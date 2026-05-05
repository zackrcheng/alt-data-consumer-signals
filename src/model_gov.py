"""
model_gov.py — DASH alt-data forecast across three target framings.

Why three targets (motivation per the modeling note in the EDA):
  • gov_yoy_growth_pct  — Total Marketplace GOV YoY. The headline number
    consensus reports. Mixes US (signals are US-centric) with international
    (Wolt + Deliveroo, structurally invisible to our features).
  • orders_yoy_growth_pct — Cleaner alignment: alt-data is a frequency proxy,
    orders IS frequency. AOV is priced by analysts, not by us.
  • gov_surprise_pct — The L/S quantity directly: % beat vs FactSet consensus.
    Stationary target; Deliveroo lives inside both sides of the surprise so it
    cancels.

Two model variants per target (per Session 9 EDA):
  drop_model — OLS on 5 features (drops dash_engagement_x_sentiment_mean, VIF=17).
  pca_model  — OLS on 3 standalone + 1 PCA composite of the collinear cluster
               {dash_engagement_x_sentiment_mean, revision_momentum_pct,
               consumer_health_index}. PCA refit per fold.

Validation: expanding-window walk-forward starting Q1 2023 (8q minimum
training). FRED `jolts_transport_yoy` is missing for Q1 2026 (publication
lag) — imputed at predict time with trailing 4-quarter mean.

Outputs (long format with `target` column):
  outputs/tables/walk_forward_predictions.csv   — per-quarter predictions
  outputs/tables/model_summary.csv              — RMSE / MAE / hit-rate
  outputs/tables/q1_2026_forecast.csv           — final point + 80% CI
  outputs/tables/model_coefficients.csv         — final-fit coef tables
  outputs/figures/walk_forward.png              — 3 subplots (one per target)
"""

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    MASTER_DF_PATH, FORECAST_QUARTER,
    MODEL_FEATURE_COLS, CORROBORATING_COLS,
    WALK_FORWARD_MIN_TRAIN_QUARTERS, WALK_FORWARD_VALIDATION_START,
    OUTPUTS_TABLES, OUTPUTS_FIGURES, RANDOM_SEED,
    CHART_STYLE, COLORS,
)


# ── Feature sets (Session 9 EDA decisions) ───────────────────────────────────
DROP_FEATURES = [
    "doordash_trends_momentum",
    "consumer_health_index",
    "prior_qtr_gov_surprise_pct",
    "revision_momentum_pct",
    "jolts_transport_yoy",
]

PCA_CLUSTER = [
    "dash_engagement_x_sentiment_mean",
    "revision_momentum_pct",
    "consumer_health_index",
]
PCA_STANDALONE = [
    "doordash_trends_momentum",
    "prior_qtr_gov_surprise_pct",
    "jolts_transport_yoy",
]
PCA_COMPOSITE_NAME = "frequency_signal_composite"

# Disjointness invariant
_all_used = set(DROP_FEATURES) | set(PCA_CLUSTER) | set(PCA_STANDALONE)
assert _all_used.issubset(set(MODEL_FEATURE_COLS)), (
    f"Features outside MODEL_FEATURE_COLS: {_all_used - set(MODEL_FEATURE_COLS)}")
assert not (_all_used & set(CORROBORATING_COLS)), (
    f"Model features collide with CORROBORATING_COLS: {_all_used & set(CORROBORATING_COLS)}")


# ── Imputation ───────────────────────────────────────────────────────────────

def _trailing_4q_mean(history: pd.Series) -> float:
    valid = history.dropna().tail(4)
    return float(valid.mean()) if len(valid) else np.nan


def _impute_row(test_row: pd.Series, train_df: pd.DataFrame,
                feature_cols: list[str]) -> pd.Series:
    out = test_row.copy()
    for c in feature_cols:
        if pd.isna(out[c]):
            out[c] = _trailing_4q_mean(train_df[c])
    return out


# ── Per-target baselines ─────────────────────────────────────────────────────

def _prior_qtr(df: pd.DataFrame, t: int, col: str) -> float:
    return float(df[col].iloc[t - 1]) if t >= 1 and pd.notna(df[col].iloc[t - 1]) else np.nan


def _trail4q(df: pd.DataFrame, t: int, col: str) -> float:
    if t < 4:
        return np.nan
    window = df[col].iloc[max(0, t - 4):t].dropna()
    return float(window.mean()) if len(window) else np.nan


def _factset_implied_yoy(df: pd.DataFrame, t: int,
                          consensus_col: str, actual_col: str) -> float:
    """YoY implied by FactSet consensus[t] vs actual[t-4]."""
    if t < 4:
        return np.nan
    cons = df[consensus_col].iloc[t]
    base = df[actual_col].iloc[t - 4]
    if pd.isna(cons) or pd.isna(base) or base == 0:
        return np.nan
    return float((cons / base - 1.0) * 100.0)


def _zero(_: pd.DataFrame, __: int) -> float:
    return 0.0


# ── Target configurations ────────────────────────────────────────────────────

TargetConfig = dict
TARGET_CONFIGS: dict[str, TargetConfig] = {
    "gov_yoy_growth_pct": {
        "label": "Total GOV YoY (%)",
        "baselines": {
            "baseline_prior":   lambda df, t: _prior_qtr(df, t, "gov_yoy_growth_pct"),
            "baseline_trail4q": lambda df, t: _trail4q(df, t, "gov_yoy_growth_pct"),
            "consensus":        lambda df, t: _factset_implied_yoy(
                df, t, "gov_factset_consensus_mn", "gov_actual_mn"),
        },
        "benchmark": "consensus",
    },
    "orders_yoy_growth_pct": {
        "label": "Orders YoY (%)",
        "baselines": {
            "baseline_prior":   lambda df, t: _prior_qtr(df, t, "orders_yoy_growth_pct"),
            "baseline_trail4q": lambda df, t: _trail4q(df, t, "orders_yoy_growth_pct"),
            "consensus":        lambda df, t: _factset_implied_yoy(
                df, t, "orders_factset_consensus_mn", "orders_actual_mn"),
        },
        "benchmark": "consensus",
    },
    "gov_surprise_pct": {
        "label": "GOV surprise vs consensus (pp)",
        "baselines": {
            # zero IS consensus by definition for a surprise target
            "baseline_zero":    _zero,
            "baseline_prior":   lambda df, t: _prior_qtr(df, t, "gov_surprise_pct"),
            "baseline_trail4q": lambda df, t: _trail4q(df, t, "gov_surprise_pct"),
        },
        "benchmark": "baseline_zero",
    },
}


# ── Models ───────────────────────────────────────────────────────────────────

def _fit_drop(train: pd.DataFrame, target: str):
    clean = train[DROP_FEATURES + [target]].dropna()
    if len(clean) < WALK_FORWARD_MIN_TRAIN_QUARTERS:
        return None
    X = sm.add_constant(clean[DROP_FEATURES])
    return sm.OLS(clean[target], X).fit()


def _predict_drop(fit, train: pd.DataFrame, test_row: pd.Series) -> float:
    test_imp = _impute_row(test_row, train, DROP_FEATURES)
    X = pd.DataFrame([test_imp[DROP_FEATURES].astype(float).values],
                     columns=DROP_FEATURES)
    X = sm.add_constant(X, has_constant="add")
    return float(fit.predict(X).iloc[0])


def _fit_pca(train: pd.DataFrame, target: str):
    clean = train[PCA_CLUSTER + PCA_STANDALONE + [target]].dropna()
    if len(clean) < WALK_FORWARD_MIN_TRAIN_QUARTERS:
        return None
    scaler = StandardScaler()
    cluster_z = scaler.fit_transform(clean[PCA_CLUSTER])
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    pc1 = pca.fit_transform(cluster_z).flatten()

    # Sign convention: PC1 loads positively on dash_engagement_x_sentiment_mean
    eng_idx = PCA_CLUSTER.index("dash_engagement_x_sentiment_mean")
    if pca.components_[0, eng_idx] < 0:
        pca.components_ *= -1.0
        pc1 = -pc1

    X = clean[PCA_STANDALONE].copy().reset_index(drop=True)
    X[PCA_COMPOSITE_NAME] = pc1
    y = clean[target].reset_index(drop=True)
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    return fit, scaler, pca


def _predict_pca(model_tuple, train: pd.DataFrame, test_row: pd.Series) -> float:
    fit, scaler, pca = model_tuple
    test_imp = _impute_row(test_row, train, PCA_CLUSTER + PCA_STANDALONE)
    cluster = pd.DataFrame([test_imp[PCA_CLUSTER].astype(float).values],
                           columns=PCA_CLUSTER)
    pc1 = pca.transform(scaler.transform(cluster)).flatten()[0]
    X = pd.DataFrame([test_imp[PCA_STANDALONE].astype(float).tolist() + [pc1]],
                     columns=PCA_STANDALONE + [PCA_COMPOSITE_NAME])
    X = sm.add_constant(X, has_constant="add")
    return float(fit.predict(X).iloc[0])


# ── Per-target walk-forward ──────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cfg = TARGET_CONFIGS[target]
    val_start_idx = df.index[df["quarter_label"] == WALK_FORWARD_VALIDATION_START][0]
    forecast_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]

    rows = []
    for t in range(val_start_idx, forecast_idx):
        actual = df[target].iloc[t]
        if pd.isna(actual):
            continue
        train = df.iloc[:t]
        test = df.iloc[t]

        row = {
            "target":           target,
            "quarter_label":    test["quarter_label"],
            "quarter_end_date": test["quarter_end_date"],
            "actual":           actual,
            "n_train":          len(train),
        }
        for name, fn in cfg["baselines"].items():
            row[name] = fn(df, t)

        drop_fit = _fit_drop(train, target)
        pca_tuple = _fit_pca(train, target)
        row["drop_model"] = _predict_drop(drop_fit, train, test) if drop_fit else np.nan
        row["pca_model"]  = _predict_pca(pca_tuple, train, test) if pca_tuple else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── Per-target evaluation ────────────────────────────────────────────────────

def evaluate(wf: pd.DataFrame, target: str) -> pd.DataFrame:
    cfg = TARGET_CONFIGS[target]
    bench = cfg["benchmark"]
    predictor_cols = [c for c in wf.columns if c not in
                      {"target", "quarter_label", "quarter_end_date", "actual", "n_train"}]
    rows = []
    for c in predictor_cols:
        cols = list({c, "actual", bench})
        sub = wf[cols].dropna()
        if sub.empty or c not in sub.columns:
            continue
        err = sub[c] - sub["actual"]
        rmse = float(np.sqrt((err ** 2).mean()))
        mae = float(err.abs().mean())
        # Hit = predictor's sign-vs-benchmark matches actual's sign-vs-benchmark.
        # For surprise target the benchmark is zero (consensus); for the others
        # the benchmark is the FactSet-implied-YoY consensus.
        actual_dir = sub["actual"] >= sub[bench]
        pred_dir = sub[c] >= sub[bench]
        hit = float((actual_dir == pred_dir).mean())
        rows.append({"target": target, "model": c, "n": len(sub),
                     "rmse_pp": rmse, "mae_pp": mae,
                     "hit_rate_vs_benchmark": hit})
    return pd.DataFrame(rows)


# ── Per-target Q1 2026 forecast with bootstrap CI ────────────────────────────

def forecast_q1_2026(df: pd.DataFrame, wf: pd.DataFrame, target: str,
                     n_boot: int = 2000) -> pd.DataFrame:
    cfg = TARGET_CONFIGS[target]
    rng = np.random.default_rng(RANDOM_SEED)
    forecast_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]
    train = df.iloc[:forecast_idx]
    test = df.iloc[forecast_idx]

    drop_fit = _fit_drop(train, target)
    pca_tuple = _fit_pca(train, target)

    drop_pt = _predict_drop(drop_fit, train, test) if drop_fit else np.nan
    pca_pt  = _predict_pca(pca_tuple, train, test) if pca_tuple else np.nan

    def _ci(point: float, residuals: np.ndarray) -> tuple[float, float]:
        if pd.isna(point) or len(residuals) == 0:
            return (np.nan, np.nan)
        sims = point + rng.choice(residuals, size=n_boot, replace=True)
        return (float(np.percentile(sims, 10)), float(np.percentile(sims, 90)))

    drop_resid = (wf["actual"] - wf["drop_model"]).dropna().values
    pca_resid = (wf["actual"] - wf["pca_model"]).dropna().values

    drop_lo, drop_hi = _ci(drop_pt, drop_resid)
    pca_lo, pca_hi   = _ci(pca_pt,  pca_resid)

    # Benchmark point estimate (the relevant consensus / zero baseline)
    bench_fn = cfg["baselines"][cfg["benchmark"]]
    bench_pt = bench_fn(df, forecast_idx)

    rows = [
        {"target": target, "model": "drop_model",
         "predicted_value": drop_pt, "ci80_lo": drop_lo, "ci80_hi": drop_hi,
         "vs_benchmark_pp": (drop_pt - bench_pt) if pd.notna(drop_pt) else np.nan,
         "benchmark_value": bench_pt, "benchmark_name": cfg["benchmark"]},
        {"target": target, "model": "pca_model",
         "predicted_value": pca_pt, "ci80_lo": pca_lo, "ci80_hi": pca_hi,
         "vs_benchmark_pp": (pca_pt - bench_pt) if pd.notna(pca_pt) else np.nan,
         "benchmark_value": bench_pt, "benchmark_name": cfg["benchmark"]},
        {"target": target, "model": cfg["benchmark"],
         "predicted_value": bench_pt, "ci80_lo": np.nan, "ci80_hi": np.nan,
         "vs_benchmark_pp": 0.0,
         "benchmark_value": bench_pt, "benchmark_name": cfg["benchmark"]},
    ]
    return pd.DataFrame(rows)


def build_coefficient_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    forecast_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]
    train = df.iloc[:forecast_idx]

    rows = []
    drop_fit = _fit_drop(train, target)
    if drop_fit is not None:
        for name in drop_fit.params.index:
            rows.append({"target": target, "model": "drop_model", "feature": name,
                         "coef": float(drop_fit.params[name]),
                         "stderr": float(drop_fit.bse[name]),
                         "p_value": float(drop_fit.pvalues[name])})
        rows.append({"target": target, "model": "drop_model", "feature": "_R_squared",
                     "coef": float(drop_fit.rsquared), "stderr": np.nan,
                     "p_value": float(drop_fit.f_pvalue)})

    pca_tuple = _fit_pca(train, target)
    if pca_tuple is not None:
        pca_fit = pca_tuple[0]
        for name in pca_fit.params.index:
            rows.append({"target": target, "model": "pca_model", "feature": name,
                         "coef": float(pca_fit.params[name]),
                         "stderr": float(pca_fit.bse[name]),
                         "p_value": float(pca_fit.pvalues[name])})
        rows.append({"target": target, "model": "pca_model", "feature": "_R_squared",
                     "coef": float(pca_fit.rsquared), "stderr": np.nan,
                     "p_value": float(pca_fit.f_pvalue)})
    return pd.DataFrame(rows)


# ── Combined plot ────────────────────────────────────────────────────────────

def plot_walk_forward(wf_all: pd.DataFrame, q1_all: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(CHART_STYLE)

    targets = list(TARGET_CONFIGS.keys())
    fig, axes = plt.subplots(len(targets), 1, figsize=(13, 4 * len(targets)),
                             sharex=False)
    for ax, target in zip(axes, targets):
        wf = wf_all[wf_all["target"] == target]
        cfg = TARGET_CONFIGS[target]
        bench = cfg["benchmark"]

        ax.plot(wf["quarter_label"], wf["actual"], "o-", lw=2.5,
                color=COLORS["actual"], label=f"Actual {cfg['label']}")
        if bench in wf.columns:
            ax.plot(wf["quarter_label"], wf[bench], "s--",
                    color=COLORS["consensus"], label=f"{bench} (benchmark)")
        ax.plot(wf["quarter_label"], wf["drop_model"], "^--", alpha=0.85,
                color=COLORS["dash_primary"], label="drop_model")
        ax.plot(wf["quarter_label"], wf["pca_model"], "v--", alpha=0.85,
                color=COLORS["forecast"], label="pca_model")

        # Q1 2026 forecast points with CI
        fc = q1_all[q1_all["target"] == target].set_index("model")
        x_fc = "Q1_2026"
        for m, color in [("drop_model", COLORS["dash_primary"]),
                         ("pca_model", COLORS["forecast"])]:
            if m not in fc.index:
                continue
            pt = fc.loc[m, "predicted_value"]
            lo, hi = fc.loc[m, "ci80_lo"], fc.loc[m, "ci80_hi"]
            if pd.notna(pt):
                ax.errorbar([x_fc], [pt],
                            yerr=[[pt - lo], [hi - pt]] if pd.notna(lo) else None,
                            fmt="*", ms=12, color=color)
        if bench in fc.index:
            bench_pt = fc.loc[bench, "predicted_value"]
            if pd.notna(bench_pt):
                ax.scatter([x_fc], [bench_pt], marker="s", s=80,
                           color=COLORS["consensus"])

        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{cfg['label']}  •  walk-forward + Q1 2026")
        ax.set_ylabel(cfg["label"])
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    df = (pd.read_csv(MASTER_DF_PATH, parse_dates=["quarter_end_date"])
            .sort_values("quarter_end_date").reset_index(drop=True))

    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    wf_frames, summary_frames, q1_frames, coef_frames = [], [], [], []
    for target in TARGET_CONFIGS:
        wf = run_walk_forward(df, target)
        wf_frames.append(wf)
        summary_frames.append(evaluate(wf, target))
        q1_frames.append(forecast_q1_2026(df, wf, target))
        coef_frames.append(build_coefficient_table(df, target))

    wf_all      = pd.concat(wf_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    q1_all      = pd.concat(q1_frames, ignore_index=True)
    coef_all    = pd.concat(coef_frames, ignore_index=True)

    wf_all.to_csv(OUTPUTS_TABLES / "walk_forward_predictions.csv", index=False)
    summary_all.to_csv(OUTPUTS_TABLES / "model_summary.csv", index=False)
    q1_all.to_csv(OUTPUTS_TABLES / "q1_2026_forecast.csv", index=False)
    coef_all.to_csv(OUTPUTS_TABLES / "model_coefficients.csv", index=False)

    plot_walk_forward(wf_all, q1_all, OUTPUTS_FIGURES / "walk_forward.png")

    for target in TARGET_CONFIGS:
        cfg = TARGET_CONFIGS[target]
        print(f"\n{'='*72}\n  {target}  ({cfg['label']})\n{'='*72}")
        print("\nWalk-forward predictions:")
        wf_t = wf_all[wf_all["target"] == target].drop(columns=["target", "quarter_end_date"])
        print(wf_t.round(2).to_string(index=False))
        print("\nSummary (lower RMSE/MAE = better; hit = sign-vs-benchmark match):")
        s = summary_all[summary_all["target"] == target].drop(columns=["target"])
        print(s.round(3).to_string(index=False))
        print("\nQ1 2026 forecast (80% CI from walk-forward residual bootstrap):")
        q = q1_all[q1_all["target"] == target].drop(columns=["target"])
        print(q.round(2).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
