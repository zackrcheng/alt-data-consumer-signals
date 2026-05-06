"""
model_crosssectional.py — UBER GB-surprise model + DASH-vs-UBER spread.

Trained on UBER data only (no pooling, per project rule). Mirrors the
model_gov.py architecture: 6 candidate features, standardized-surprise
target via causal expanding-std, walk-forward expanding window starting
at MIN_TRAIN_QUARTERS=8, six architectures (ols_drop, pca, pls, ridge,
lasso, ols_1feat), bootstrap + QuantReg 80% CI.

UBER-side feature set parallel to DASH:
  ubereats_trends_momentum            (analog to doordash_trends_momentum)
  uber_engagement_x_sentiment_mean    (analog to dash_engagement_x_sentiment)
  consumer_health_index               (shared macro)
  prior_qtr_uber_gb_surprise_pct      (autoregressive, mirrors prior surprise)
  uber_revision_momentum_pct          (UBER IBES analyst revisions)
  jolts_transport_yoy                 (shared macro)

Target:
  TARGET_RAW = gb_total_surprise_pct   (UBER FactSet Total-GB surprise)
  TARGET_STD = uber_gb_surprise_std    (causal expanding std-normalized)

The cross-sectional spread is the L/S signal for the long-DASH/short-UBER
or sector-tilt framing:
  spread = DASH_predicted_surprise − UBER_predicted_surprise

Conservative spread CI:
  ci_lo = dash_ci_lo − uber_ci_hi    (worst-case for long-DASH)
  ci_hi = dash_ci_hi − uber_ci_lo    (best-case)

Outputs:
  outputs/tables/uber_model_comparison.csv     all UBER variants ranked
  outputs/tables/cross_sectional_spread.csv    DASH vs UBER spread + signal
  outputs/figures/dash_vs_uber_forecast.png    side-by-side forecast plot
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from src.config import (
    UBER_GOV_MASTER_PATH, MASTER_DF_PATH, GOOGLE_TRENDS_PATH, APPSTORE_PATH,
    IBES_CONSENSUS_PATH, FORECAST_QUARTER, OUTPUTS_TABLES, OUTPUTS_FIGURES,
    RANDOM_SEED, TRENDS_WINDOW_WEEKS, TRENDS_LAG_WEEKS, QUARTER_END_DATES,
    CHART_STYLE, COLORS,
)
from src.utils import weekly_to_quarterly
from src.model_gov import (
    OLSDrop, PCAModel, PLSModel, RidgeModel, LassoModel, OLS1Feat,
    MIN_TRAIN_QUARTERS, MIN_VALID_TRAIN_ROWS, MIN_VALID_FOR_TOP_PICK,
    N_BOOTSTRAP, run_walk_forward, evaluate, predict_q1_2026,
)


# ── UBER feature set + targets ──────────────────────────────────────────────

UBER_FEATURES = [
    "ubereats_trends_momentum",
    "uber_engagement_x_sentiment_mean",
    "consumer_health_index",
    "prior_qtr_uber_gb_surprise_pct",
    "uber_revision_momentum_pct",
    "jolts_transport_yoy",
]

UBER_TARGET_RAW = "gb_total_surprise_pct"
UBER_TARGET_STD = "uber_gb_surprise_std"

UBER_MODEL_CLASSES = {
    "ols_drop":  OLSDrop,
    "pca":       PCAModel,
    "pls":       PLSModel,
    "ridge":     RidgeModel,
    "lasso":     LassoModel,
    "ols_1feat": OLS1Feat,
}


# ── UBER feature engineering ────────────────────────────────────────────────

def build_uber_master() -> pd.DataFrame:
    """Build the UBER-side feature DataFrame parallel to DASH master_df.

    Pulls each feature from its native source and aligns to the UBER GOV
    spine (Q4 2020 → Q1 2026). All look-ahead-safe."""
    uber = pd.read_csv(UBER_GOV_MASTER_PATH, parse_dates=["quarter_end_date"])
    uber = uber.sort_values("quarter_end_date").reset_index(drop=True)

    # Standardized target (causal expanding std, .shift(1))
    s = uber[UBER_TARGET_RAW]
    expanding_std = s.expanding(min_periods=4).std().shift(1)
    uber[UBER_TARGET_STD] = s / expanding_std
    uber["_uber_expanding_std"] = expanding_std

    # Autoregressive feature
    uber["prior_qtr_uber_gb_surprise_pct"] = uber[UBER_TARGET_RAW].shift(1)

    # Trends — 8-week mean of "Uber Eats" weekly index, with QoQ momentum
    trends = pd.read_csv(GOOGLE_TRENDS_PATH, parse_dates=["date"])
    if "Uber Eats" in trends.columns:
        ut = weekly_to_quarterly(trends, "date", "Uber Eats",
                                  TRENDS_WINDOW_WEEKS, TRENDS_LAG_WEEKS)
        ut = ut.rename(columns={"Uber Eats_mean": "ubereats_trends_mean"})
        uber = uber.merge(ut[["quarter_label", "ubereats_trends_mean"]],
                           on="quarter_label", how="left")
        # Recompute sort then take pct_change (causal: only past values)
        uber = uber.sort_values("quarter_end_date").reset_index(drop=True)
        uber["ubereats_trends_momentum"] = (
            uber["ubereats_trends_mean"].pct_change(fill_method=None) * 100
        )

    # AppStore engagement × sentiment for UBER (weekly → 8-week mean)
    appstore = pd.read_csv(APPSTORE_PATH, parse_dates=["date"])
    if "uber_engagement_x_sentiment" in appstore.columns:
        ua = weekly_to_quarterly(appstore, "date", "uber_engagement_x_sentiment",
                                  TRENDS_WINDOW_WEEKS, TRENDS_LAG_WEEKS)
        # column is already named uber_engagement_x_sentiment_mean
        uber = uber.merge(ua[["quarter_label", "uber_engagement_x_sentiment_mean"]],
                           on="quarter_label", how="left")

    # Macro features — pull from already-aggregated DASH master_df
    # (consumer_health_index and jolts_transport_yoy are not DASH-specific —
    # they're macro columns shared across both companies)
    master = pd.read_csv(MASTER_DF_PATH)
    macro_cols = ["quarter_label", "consumer_health_index", "jolts_transport_yoy"]
    uber = uber.merge(master[macro_cols], on="quarter_label", how="left")

    # IBES UBER (already at quarterly grain)
    ibes = pd.read_csv(IBES_CONSENSUS_PATH)
    ibes_u = ibes[ibes["ticker"] == "UBER"][[
        "quarter_label", "revision_momentum_pct"]].rename(
        columns={"revision_momentum_pct": "uber_revision_momentum_pct"})
    uber = uber.merge(ibes_u, on="quarter_label", how="left")

    return uber.sort_values("quarter_end_date").reset_index(drop=True)


def impute_forecast_row_uber(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing-4q-mean impute any NaN in the Q1 2026 forecast row's
    UBER feature values. Mirrors the DASH-side jolts imputation rule."""
    df = df.copy()
    fc_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]
    hist = df.iloc[:fc_idx]
    for c in UBER_FEATURES:
        if c not in df.columns:
            continue
        if pd.isna(df.at[fc_idx, c]):
            tail = hist[c].dropna().tail(4)
            if len(tail):
                df.at[fc_idx, c] = float(tail.mean())
    return df


# ── Cross-sectional spread ──────────────────────────────────────────────────

def compute_spread(prereg_dash: pd.Series, top_uber: pd.Series) -> dict:
    """L/S spread = DASH_pred − UBER_pred, with conservative CI bounds."""
    dash_pt = float(prereg_dash["q1_2026_pred_pct"])
    dash_lo = float(prereg_dash["q1_2026_ci_80_lo"])
    dash_hi = float(prereg_dash["q1_2026_ci_80_hi"])
    uber_pt = float(top_uber["q1_2026_pred_pct"])
    uber_lo = float(top_uber["q1_2026_ci_80_lo"])
    uber_hi = float(top_uber["q1_2026_ci_80_hi"])

    spread_pt = dash_pt - uber_pt
    # Conservative: worst-case for long-DASH (DASH at low, UBER at high)
    spread_ci_lo = dash_lo - uber_hi
    spread_ci_hi = dash_hi - uber_lo

    if spread_ci_lo > 0:
        signal = "LONG_DASH / FADE_UBER  (spread CI strictly positive)"
    elif spread_pt > 0:
        signal = "lean LONG DASH (spread positive but CI straddles 0)"
    elif spread_pt < 0:
        signal = "lean FADE DASH (spread negative)"
    else:
        signal = "NEUTRAL"

    return {
        "dash_top_variant":     prereg_dash["selected_variant"],
        "uber_top_variant":     top_uber["variant_name"],
        "dash_pred_pct":        dash_pt,
        "dash_ci80_lo":         dash_lo,
        "dash_ci80_hi":         dash_hi,
        "uber_pred_pct":        uber_pt,
        "uber_ci80_lo":         uber_lo,
        "uber_ci80_hi":         uber_hi,
        "spread_dash_minus_uber_pct": spread_pt,
        "spread_ci80_lo_conservative": spread_ci_lo,
        "spread_ci80_hi_conservative": spread_ci_hi,
        "ls_signal":            signal,
    }


# ── Plot ────────────────────────────────────────────────────────────────────

def plot_dash_vs_uber(spread: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(CHART_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Side-by-side forecast
    ax = axes[0]
    names = ["DASH\n(GOV surprise)", "UBER\n(Total-GB surprise)"]
    points = [spread["dash_pred_pct"], spread["uber_pred_pct"]]
    los = [spread["dash_ci80_lo"], spread["uber_ci80_lo"]]
    his = [spread["dash_ci80_hi"], spread["uber_ci80_hi"]]
    cols = [COLORS["dash_primary"], COLORS["uber"] if "uber" in COLORS else "black"]
    for i, (name, pt, lo, hi, c) in enumerate(zip(names, points, los, his, cols)):
        ax.errorbar([i], [pt], yerr=[[max(0, pt-lo)], [max(0, hi-pt)]],
                    fmt="*", ms=18, color=c, capsize=8,
                    label=f"{name.split(chr(10))[0]}: {pt:+.2f}pp [{lo:+.1f}, {hi:+.1f}]")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Q1 2026 surprise (pp)")
    ax.set_title("DASH vs UBER — Q1 2026 surprise forecasts")
    ax.legend(fontsize=8, loc="best")

    # (b) Spread bar with CI
    ax = axes[1]
    spread_pt = spread["spread_dash_minus_uber_pct"]
    spread_lo = spread["spread_ci80_lo_conservative"]
    spread_hi = spread["spread_ci80_hi_conservative"]
    bar_color = (COLORS["actual"] if spread_lo > 0 else
                  (COLORS["forecast"] if spread_pt > 0 else COLORS["dash_primary"]))
    ax.barh([0], [spread_pt], color=bar_color, alpha=0.7, height=0.4)
    ax.errorbar([spread_pt], [0], xerr=[[max(0, spread_pt-spread_lo)],
                                          [max(0, spread_hi-spread_pt)]],
                fmt="o", ms=10, color="black", capsize=8)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks([0]); ax.set_yticklabels(["DASH − UBER\nspread"], fontsize=10)
    ax.set_xlabel("Q1 2026 surprise spread (pp)")
    ax.set_title(f"Cross-sectional spread:  {spread['ls_signal']}\n"
                  f"point = {spread_pt:+.2f}pp · 80% CI [{spread_lo:+.2f}, {spread_hi:+.2f}]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    df = build_uber_master()
    df = impute_forecast_row_uber(df)

    print("=" * 72)
    print("UBER feature coverage on historical sample")
    print("=" * 72)
    hist = df[df["quarter_label"] != FORECAST_QUARTER]
    for c in UBER_FEATURES:
        n = df[c].notna().sum() if c in df.columns else 0
        nh = hist[c].notna().sum() if c in hist.columns else 0
        print(f"  {c:38s}  {nh}/{len(hist)} hist  (Q1 2026: "
              f"{'✓' if c in df.columns and pd.notna(df.loc[df['quarter_label']==FORECAST_QUARTER, c].iloc[0]) else '✗'})")

    # Forecast-time conversion factor for std target
    surprise_hist = hist[UBER_TARGET_RAW].dropna()
    forecast_uber_std = (
        float(surprise_hist.expanding(min_periods=4).std().iloc[-1])
        if len(surprise_hist) >= 4 else np.nan
    )
    print(f"\nUBER forecast-time expanding std (for unstandardizing): "
          f"{forecast_uber_std:.3f}pp")

    # Run all variants
    targets = [UBER_TARGET_RAW, UBER_TARGET_STD]
    rows = []
    for target in targets:
        for model_name, model_cls in UBER_MODEL_CLASSES.items():
            variant = f"current__{target}__{model_name}"
            try:
                wf = run_walk_forward(df, UBER_FEATURES, target, model_cls)
                metrics = evaluate(wf)
                fcst = predict_q1_2026(df, UBER_FEATURES, target, model_cls)
                if target == UBER_TARGET_STD:
                    pred_pct = (fcst["point"] * forecast_uber_std
                                if pd.notna(fcst["point"]) else np.nan)
                    ci_lo_pct = (fcst["ci_lo"] * forecast_uber_std
                                 if pd.notna(fcst["ci_lo"]) else np.nan)
                    ci_hi_pct = (fcst["ci_hi"] * forecast_uber_std
                                 if pd.notna(fcst["ci_hi"]) else np.nan)
                else:
                    pred_pct, ci_lo_pct, ci_hi_pct = (
                        fcst["point"], fcst["ci_lo"], fcst["ci_hi"])
                rows.append({
                    "variant_name": variant, "target": target,
                    "model": model_name, **metrics,
                    "q1_2026_pred_raw": fcst["point"],
                    "q1_2026_pred_pct": pred_pct,
                    "q1_2026_ci_80_lo": ci_lo_pct,
                    "q1_2026_ci_80_hi": ci_hi_pct,
                })
            except Exception as e:
                print(f"  {variant} FAILED: {e}")

    comp = pd.DataFrame(rows).sort_values(
        ["directional_acc", "rmse"], ascending=[False, True]).reset_index(drop=True)
    comp.to_csv(OUTPUTS_TABLES / "uber_model_comparison.csv", index=False)

    print()
    print("=" * 72)
    print("UBER MODEL COMPARISON")
    print("=" * 72)
    print(comp.round(3).to_string(index=False))

    # Top variant — same eligibility filter as DASH
    eligible = comp[comp["n_valid"] >= MIN_VALID_FOR_TOP_PICK]
    if eligible.empty:
        print(f"\nNo eligible UBER variants (n_valid >= {MIN_VALID_FOR_TOP_PICK})"
              "  — falling back to highest-ranked variant.")
        eligible = comp
    top_uber = eligible.iloc[0]
    print(f"\nTop UBER variant: {top_uber['variant_name']}")
    print(f"  directional_acc = {top_uber['directional_acc']:.3f}")
    print(f"  rmse            = {top_uber['rmse']:.3f}")
    print(f"  n_valid         = {int(top_uber['n_valid'])}")
    print(f"  Q1 2026 pred    = {top_uber['q1_2026_pred_pct']:+.2f}pp  "
          f"(80% CI [{top_uber['q1_2026_ci_80_lo']:+.2f}, {top_uber['q1_2026_ci_80_hi']:+.2f}])")

    # Cross-sectional spread
    prereg_dash = pd.read_csv(OUTPUTS_TABLES / "q1_2026_preregistered.csv").iloc[0]
    spread = compute_spread(prereg_dash, top_uber)
    spread_df = pd.DataFrame([spread])
    spread_df.to_csv(OUTPUTS_TABLES / "cross_sectional_spread.csv", index=False)

    print()
    print("=" * 72)
    print("CROSS-SECTIONAL SPREAD — DASH vs UBER Q1 2026 SURPRISE")
    print("=" * 72)
    print(f"  DASH ({spread['dash_top_variant']}):")
    print(f"    Q1 2026 surprise = {spread['dash_pred_pct']:+.2f}pp  "
          f"(80% CI [{spread['dash_ci80_lo']:+.2f}, {spread['dash_ci80_hi']:+.2f}])")
    print(f"  UBER ({spread['uber_top_variant']}):")
    print(f"    Q1 2026 surprise = {spread['uber_pred_pct']:+.2f}pp  "
          f"(80% CI [{spread['uber_ci80_lo']:+.2f}, {spread['uber_ci80_hi']:+.2f}])")
    print(f"  Spread (DASH − UBER):")
    print(f"    point = {spread['spread_dash_minus_uber_pct']:+.2f}pp")
    print(f"    conservative 80% CI = [{spread['spread_ci80_lo_conservative']:+.2f}, "
          f"{spread['spread_ci80_hi_conservative']:+.2f}]")
    print(f"    L/S signal: {spread['ls_signal']}")

    plot_dash_vs_uber(spread, OUTPUTS_FIGURES / "dash_vs_uber_forecast.png")
    print(f"\nSaved: outputs/figures/dash_vs_uber_forecast.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
