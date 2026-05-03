"""
model_transmission.py — pass-through sensitivity regressions and variance decomposition.

Three OLS regressions that quantify the causal chain from GOV to stock price:
  1. GOV growth → Revenue growth (take rate pass-through)
  2. Revenue surprise → EBITDA margin change (operating leverage)
  3. GOV surprise → CAR[-1,+2] (event study link)

Also runs variance decomposition of GOV to justify feature selection.

DASH + CART panel used for transmission regressions only (more observations).
Never use this panel to forecast DASH GOV.

See CLAUDE.md §12 for full spec.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

from src.config import (
    MASTER_DF_PATH, COMPUSTAT_PATH, CRSP_EVENT_STUDY_PATH,
    OUTPUTS_TABLES, RANDOM_SEED,
)
from src.utils import print_pull_summary


def regression_gov_to_revenue(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    GOV growth → Revenue growth (DASH only).
    Expected β ≈ 0.90–0.95 (stable take rate implies near 1:1 pass-through).
    """
    avail = df.dropna(subset=["gov_yoy_growth_pct", "revenue_actual_bn"]).copy()
    avail["revenue_yoy_pct"] = avail["revenue_actual_bn"].pct_change(4) * 100
    avail = avail.dropna(subset=["revenue_yoy_pct"])

    X = sm.add_constant(avail[["gov_yoy_growth_pct"]])
    y = avail["revenue_yoy_pct"]
    model = sm.OLS(y, X).fit()

    print("\nRegression 1: GOV growth → Revenue growth (DASH)")
    print(f"  β (pass-through): {model.params.get('gov_yoy_growth_pct', 'n/a'):.3f}")
    print(f"  R²: {model.rsquared:.3f}  |  n: {int(model.nobs)}")
    return model


def regression_revenue_to_ebitda(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Revenue growth surprise → EBITDA margin change (DASH only).
    Shows fixed cost operating leverage.
    """
    avail = df.dropna(subset=["revenue_actual_bn", "ebitda_actual_bn"]).copy()
    avail["revenue_yoy_pct"] = avail["revenue_actual_bn"].pct_change(4) * 100
    avail["ebitda_margin_pct"] = (avail["ebitda_actual_bn"] / avail["revenue_actual_bn"]) * 100
    avail["ebitda_margin_change_pp"] = avail["ebitda_margin_pct"].diff()

    # Revenue surprise = actual - consensus estimate
    if "revenue_consensus_est_bn" in avail.columns:
        avail["revenue_surprise_pct"] = (
            (avail["revenue_actual_bn"] - avail["revenue_consensus_est_bn"])
            / avail["revenue_consensus_est_bn"] * 100
        )
        x_col = "revenue_surprise_pct"
    else:
        x_col = "revenue_yoy_pct"

    avail = avail.dropna(subset=[x_col, "ebitda_margin_change_pp"])
    X = sm.add_constant(avail[[x_col]])
    y = avail["ebitda_margin_change_pp"]
    model = sm.OLS(y, X).fit()

    print("\nRegression 2: Revenue surprise → EBITDA margin change (DASH)")
    print(f"  β (operating leverage): {model.params.get(x_col, 'n/a'):.3f}")
    print(f"  R²: {model.rsquared:.3f}  |  n: {int(model.nobs)}")
    return model


def regression_gov_surprise_to_car(event_path: Path = CRSP_EVENT_STUDY_PATH) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    GOV surprise % → CAR[-1,+2] (from CRSP event study).
    Requires event_study.py to have already computed and saved CARs.
    """
    try:
        df = pd.read_csv(event_path)
    except FileNotFoundError:
        print(f"  Warning: {event_path} not found. Run event_study.py first.")
        return None

    avail = df.dropna(subset=["gov_surprise_pct", "car_minus1_plus2"])
    if len(avail) < 3:
        print("  Insufficient observations for GOV surprise → CAR regression.")
        return None

    X = sm.add_constant(avail[["gov_surprise_pct"]])
    y = avail["car_minus1_plus2"]
    model = sm.OLS(y, X).fit()

    print("\nRegression 3: GOV surprise → CAR[-1,+2] (event study)")
    print(f"  β: {model.params.get('gov_surprise_pct', 'n/a'):.3f}")
    print(f"  R²: {model.rsquared:.3f}  |  n: {int(model.nobs)}")
    return model


def variance_decomposition_gov(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose historical GOV YoY variance into feature groups via sequential R².
    Features: trends_momentum, consumer_health_index, prior_surprise, take_rate_trend.

    Returns DataFrame with feature, R² (sequential), marginal R², % variance explained.
    """
    target = "gov_yoy_growth_pct"
    feature_groups = [
        ("doordash_trends_momentum", "Trends momentum"),
        ("consumer_health_index", "Consumer health"),
        ("prior_qtr_gov_surprise_pct", "Prior surprise"),
        ("prior_qtr_take_rate", "Take rate trend"),
    ]

    avail_features = [f for f, _ in feature_groups if f in df.columns]
    avail = df.dropna(subset=[target] + avail_features).copy()

    records = []
    cumulative_r2 = 0.0

    for feat, label in feature_groups:
        if feat not in df.columns:
            continue
        features_so_far = [f for f, _ in feature_groups[:feature_groups.index((feat, label)) + 1]
                           if f in df.columns]
        X = sm.add_constant(avail[features_so_far])
        y = avail[target]
        model = sm.OLS(y, X).fit()
        marginal_r2 = model.rsquared - cumulative_r2
        records.append({
            "feature": label,
            "col": feat,
            "cumulative_r2": round(model.rsquared, 3),
            "marginal_r2": round(marginal_r2, 3),
            "pct_variance_explained": round(marginal_r2 * 100, 1),
        })
        cumulative_r2 = model.rsquared

    result = pd.DataFrame(records)
    print("\nVariance decomposition of GOV YoY growth:")
    print(result.to_string(index=False))
    return result


def run_all_transmission(df: pd.DataFrame = None) -> dict:
    """Run all transmission mechanism regressions and return results dict."""
    if df is None:
        df = pd.read_csv(MASTER_DF_PATH, parse_dates=["quarter_end_date"])

    results = {}
    results["gov_to_revenue"] = regression_gov_to_revenue(df)
    results["revenue_to_ebitda"] = regression_revenue_to_ebitda(df)
    results["gov_surprise_to_car"] = regression_gov_surprise_to_car()
    results["variance_decomp"] = variance_decomposition_gov(df)

    # Save variance decomposition table
    if isinstance(results["variance_decomp"], pd.DataFrame):
        results["variance_decomp"].to_csv(
            OUTPUTS_TABLES / "variance_decomposition.csv", index=False
        )
    return results


if __name__ == "__main__":
    run_all_transmission()
