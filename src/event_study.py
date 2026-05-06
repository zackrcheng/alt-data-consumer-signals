"""
event_study.py — DASH earnings CAR study + β3 regression.

Computes CAR[-1,+2] and CAR[0,+1] around each DASH earnings event using:
  1. CRSP daily abnormal returns (data/raw/crsp_event_study.csv) for events
     within CRSP coverage (currently through 2024-12-31).
  2. yfinance + SPY fallback for events after CRSP cutoff (per project rule §11).

β3:  CAR[-1,+2] ~ gov_surprise_pct
Applied to the pre-registered Q1 2026 GOV surprise → expected abnormal
return on May 6 2026 earnings.

Outputs:
  outputs/tables/event_study_cars.csv      per-event CAR + surprise + source
  outputs/tables/event_study_beta3.csv     β3 regression + Q1 2026 application
  outputs/figures/event_study_scatter.png  CAR vs GOV-surprise scatter + Q1 2026
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    EARNINGS_DATES, CAR_WINDOWS,
    CRSP_EVENT_STUDY_PATH, PRICES_DAILY_PATH, MASTER_DF_PATH,
    OUTPUTS_TABLES, OUTPUTS_FIGURES, COLORS, CHART_STYLE,
)


# ── Load helpers ────────────────────────────────────────────────────────────

def load_crsp_data() -> pd.DataFrame:
    if not Path(CRSP_EVENT_STUDY_PATH).exists():
        return pd.DataFrame()
    df = pd.read_csv(CRSP_EVENT_STUDY_PATH, parse_dates=["date"])
    return df


def load_yfinance_for_event_study(ticker: str = "DASH",
                                    benchmark: str = "SPY") -> pd.DataFrame:
    """yfinance daily prices → abnormal return = ticker − benchmark."""
    df = pd.read_csv(PRICES_DAILY_PATH, parse_dates=["date"])
    pivot = df.pivot(index="date", columns="ticker", values="daily_return_pct")
    if ticker not in pivot.columns or benchmark not in pivot.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date":              pivot.index,
        "ret_stock":         pivot[ticker] / 100.0,        # daily_return_pct is %
        "ret_market_spy":    pivot[benchmark] / 100.0,
    })
    out["abnormal_return_yf"] = out["ret_stock"] - out["ret_market_spy"]
    return out.reset_index(drop=True).sort_values("date")


# ── CAR computation ─────────────────────────────────────────────────────────

def _car_for_window(daily: pd.DataFrame, abnormal_col: str,
                    event_date: pd.Timestamp,
                    window: tuple[int, int]) -> tuple[float, str]:
    """CAR over a (start, end) trading-day offset window centered at event_date.
    Returns (car, status) where status ∈ {"ok", "out_of_coverage", "incomplete"}."""
    daily = daily.sort_values("date").reset_index(drop=True)
    if (daily["date"] == event_date).any():
        idx = daily.index[daily["date"] == event_date][0]
    else:
        future = daily[daily["date"] >= event_date]
        if future.empty:
            return np.nan, "out_of_coverage"
        idx = future.index[0]
    lo, hi = window
    start_i = max(0, idx + lo)
    end_i = min(len(daily) - 1, idx + hi)
    abn = daily[abnormal_col].iloc[start_i:end_i + 1]
    if abn.isna().any() or len(abn) < (hi - lo + 1):
        return np.nan, "incomplete"
    return float(abn.sum()), "ok"


def build_event_table(ticker: str = "DASH") -> pd.DataFrame:
    """Per-event CAR table for one ticker. Hybrid CRSP + yfinance source."""
    crsp = load_crsp_data()
    yfin = load_yfinance_for_event_study(ticker=ticker)
    crsp_t = crsp[crsp["ticker"] == ticker].copy() if not crsp.empty else pd.DataFrame()

    rows = []
    for quarter, event_date_str in EARNINGS_DATES.items():
        event_date = pd.Timestamp(event_date_str)
        car_p_crsp = car_t_crsp = (np.nan, "no_crsp")
        car_p_yf = car_t_yf = (np.nan, "no_yf")

        # Try CRSP first — only if the event window is within CRSP coverage
        if not crsp_t.empty:
            crsp_max = crsp_t["date"].max()
            if event_date + pd.Timedelta(days=10) <= crsp_max:
                car_p_crsp = _car_for_window(crsp_t, "abnormal_return",
                                              event_date, CAR_WINDOWS["primary"])
                car_t_crsp = _car_for_window(crsp_t, "abnormal_return",
                                              event_date, CAR_WINDOWS["tight"])

        # yfinance fallback
        if not yfin.empty:
            car_p_yf = _car_for_window(yfin, "abnormal_return_yf",
                                        event_date, CAR_WINDOWS["primary"])
            car_t_yf = _car_for_window(yfin, "abnormal_return_yf",
                                        event_date, CAR_WINDOWS["tight"])

        if car_p_crsp[1] == "ok":
            car_p, car_t, src = car_p_crsp[0], car_t_crsp[0], "CRSP"
        elif car_p_yf[1] == "ok":
            car_p, car_t, src = car_p_yf[0], car_t_yf[0], "yfinance_SPY"
        else:
            car_p = car_t = np.nan
            src = "missing"

        rows.append({
            "ticker":               ticker,
            "quarter_label":        quarter,
            "earnings_date":        event_date_str,
            "car_minus1_plus2_pct": (car_p * 100) if pd.notna(car_p) else np.nan,
            "car_0_plus1_pct":      (car_t * 100) if pd.notna(car_t) else np.nan,
            "source":               src,
        })

    return pd.DataFrame(rows)


# ── β3 regression ───────────────────────────────────────────────────────────

def fit_beta3(event_df: pd.DataFrame, master_df: pd.DataFrame,
                target_col: str = "car_minus1_plus2_pct") -> dict:
    """β3: CAR ~ gov_surprise_pct."""
    merged = event_df.merge(
        master_df[["quarter_label", "gov_surprise_pct"]],
        on="quarter_label", how="left",
    ).dropna(subset=[target_col, "gov_surprise_pct"])

    X = sm.add_constant(merged[["gov_surprise_pct"]])
    y = merged[target_col]
    fit = sm.OLS(y, X).fit()
    return {
        "beta3":     float(fit.params["gov_surprise_pct"]),
        "stderr":    float(fit.bse["gov_surprise_pct"]),
        "p_value":   float(fit.pvalues["gov_surprise_pct"]),
        "ci95_lo":   float(fit.conf_int(alpha=0.05).loc["gov_surprise_pct", 0]),
        "ci95_hi":   float(fit.conf_int(alpha=0.05).loc["gov_surprise_pct", 1]),
        "intercept": float(fit.params["const"]),
        "r_squared": float(fit.rsquared),
        "n":         int(fit.nobs),
        "fit":       fit,
        "merged":    merged,
        "target":    target_col,
    }


def apply_beta3_to_q1_2026(beta3: dict, prereg: pd.Series) -> dict:
    """Propagate the published Q1 2026 GOV surprise + 80% CI through β3."""
    gov_pt = float(prereg["q1_2026_pred_pct"])
    gov_lo = float(prereg["q1_2026_ci_80_lo"])
    gov_hi = float(prereg["q1_2026_ci_80_hi"])

    car_pt = beta3["intercept"] + beta3["beta3"] * gov_pt
    car_a  = beta3["intercept"] + beta3["beta3"] * gov_lo
    car_b  = beta3["intercept"] + beta3["beta3"] * gov_hi
    car_lo, car_hi = (car_a, car_b) if car_a <= car_b else (car_b, car_a)
    return {
        "gov_surprise_pp":      gov_pt,
        "gov_surprise_ci80_lo": gov_lo,
        "gov_surprise_ci80_hi": gov_hi,
        "expected_car_pct":     car_pt,
        "expected_car_ci80_lo": car_lo,
        "expected_car_ci80_hi": car_hi,
    }


# ── Plot ────────────────────────────────────────────────────────────────────

def plot_event_study(beta3: dict, applied: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(CHART_STYLE)

    merged = beta3["merged"]
    fig, ax = plt.subplots(figsize=(11, 6))

    crsp_pts = merged[merged["source"] == "CRSP"]
    yfin_pts = merged[merged["source"] == "yfinance_SPY"]
    if not crsp_pts.empty:
        ax.scatter(crsp_pts["gov_surprise_pct"], crsp_pts["car_minus1_plus2_pct"],
                   s=80, color=COLORS["dash_primary"],
                   label=f"CRSP (n={len(crsp_pts)})", zorder=3)
    if not yfin_pts.empty:
        ax.scatter(yfin_pts["gov_surprise_pct"], yfin_pts["car_minus1_plus2_pct"],
                   s=80, color=COLORS["forecast"], marker="^",
                   label=f"yfinance fallback (n={len(yfin_pts)})", zorder=3)

    for _, r in merged.iterrows():
        ax.annotate(r["quarter_label"].replace("_", " "),
                    (r["gov_surprise_pct"], r["car_minus1_plus2_pct"]),
                    fontsize=7, alpha=0.6,
                    xytext=(4, 2), textcoords="offset points")

    x_min = merged["gov_surprise_pct"].min() - 0.5
    x_max = max(merged["gov_surprise_pct"].max(),
                 applied["gov_surprise_ci80_hi"]) + 0.5
    xs = np.linspace(x_min, x_max, 50)
    ax.plot(xs, beta3["intercept"] + beta3["beta3"] * xs,
            color=COLORS["actual"], lw=2,
            label=f"β3 = {beta3['beta3']:+.2f}  "
                  f"(p={beta3['p_value']:.3f}, R²={beta3['r_squared']:.2f}, n={beta3['n']})")

    pt = applied["expected_car_pct"]
    lo, hi = applied["expected_car_ci80_lo"], applied["expected_car_ci80_hi"]
    g_pt = applied["gov_surprise_pp"]
    g_lo, g_hi = applied["gov_surprise_ci80_lo"], applied["gov_surprise_ci80_hi"]
    ax.errorbar([g_pt], [pt],
                xerr=[[g_pt - g_lo], [g_hi - g_pt]],
                yerr=[[max(0, pt - lo)], [max(0, hi - pt)]],
                fmt="*", ms=18, color=COLORS["consensus"], capsize=8,
                label=f"Q1 2026 expected: GOV {g_pt:+.2f}pp → CAR {pt:+.2f}%")

    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("GOV surprise vs FactSet consensus (pp)")
    ax.set_ylabel("CAR[-1, +2] (%)")
    ax.set_title("β3: stock reaction to GOV surprise — DASH earnings events")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    event_df = build_event_table(ticker="DASH")
    event_df.to_csv(OUTPUTS_TABLES / "event_study_cars.csv", index=False)

    print("=" * 72)
    print("DASH EARNINGS EVENT STUDY — CARs")
    print("=" * 72)
    print(event_df.to_string(index=False))
    print(f"\nSource breakdown: {event_df['source'].value_counts().to_dict()}")

    master = pd.read_csv(MASTER_DF_PATH)
    beta3 = fit_beta3(event_df, master)

    print()
    print("=" * 72)
    print("β3: CAR[-1,+2] ~ gov_surprise_pct")
    print("=" * 72)
    print(f"  β3            = {beta3['beta3']:+.3f}")
    print(f"  stderr        = {beta3['stderr']:.3f}")
    print(f"  p-value       = {beta3['p_value']:.3f}")
    print(f"  95% CI         = [{beta3['ci95_lo']:+.3f}, {beta3['ci95_hi']:+.3f}]")
    print(f"  R²             = {beta3['r_squared']:.3f}")
    print(f"  n              = {beta3['n']}")
    print(f"  intercept     = {beta3['intercept']:+.3f}")

    prereg = pd.read_csv(OUTPUTS_TABLES / "q1_2026_preregistered.csv").iloc[0]
    applied = apply_beta3_to_q1_2026(beta3, prereg)

    print()
    print("=" * 72)
    print("Q1 2026 EXPECTED CAR (β3 applied to pre-registered GOV surprise)")
    print("=" * 72)
    print(f"  GOV surprise (model):     {applied['gov_surprise_pp']:+.2f}pp  "
          f"(80% CI [{applied['gov_surprise_ci80_lo']:+.2f}, "
          f"{applied['gov_surprise_ci80_hi']:+.2f}])")
    print(f"  → expected CAR[-1,+2]:    {applied['expected_car_pct']:+.2f}%  "
          f"(80% CI [{applied['expected_car_ci80_lo']:+.2f}, "
          f"{applied['expected_car_ci80_hi']:+.2f}])")

    out_dict = {**{f"beta3_{k}": v for k, v in beta3.items()
                    if k not in ("fit", "merged")},
                **applied}
    pd.DataFrame([out_dict]).to_csv(
        OUTPUTS_TABLES / "event_study_beta3.csv", index=False)

    plot_event_study(beta3, applied, OUTPUTS_FIGURES / "event_study_scatter.png")
    print(f"\nSaved: outputs/figures/event_study_scatter.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
