"""
model_crosssectional.py — UBER independent GOV model for cross-sectional comparison.

TRAINED ON UBER DATA ONLY. Identical OLS architecture to model_gov.py.
Never mix DASH training data here — model independence is strictly enforced.

Purpose: compare Q1 2026 forecasts for DASH vs. UBER to generate
Long DASH / Short UBER signal (Option C framing, CLAUDE.md §2).

Output: if DASH is above-consensus and UBER is at/below → single-name long signal.
        if both same direction → sector view, weaker conviction.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

from src.config import (
    UBER_GOV_MASTER_PATH, GOOGLE_TRENDS_PATH, FRED_MACRO_PATH,
    FORECAST_QUARTER, OUTPUTS_TABLES, OLS_BASE_FEATURES, RANDOM_SEED,
)
from src.utils import weekly_to_quarterly, monthly_to_quarterly, validate_no_lookahead


# UBER-specific Trends features (parallel to DASH features in OLS_BASE_FEATURES)
UBER_TRENDS_FEATURES = [
    "uber_trends_momentum",       # QoQ change in Uber Eats Trends index
    "uber_three_way_share_mean",  # Uber Eats / (DoorDash + Uber Eats + Instacart)
]

UBER_MODEL_FEATURES = [
    "uber_trends_momentum",
    "uber_three_way_share_mean",
    "consumer_health_index",      # same macro feature as DASH model (shared)
    "uber_prior_gov_surprise_pct",
]

UBER_MODEL_TARGET = "gov_yoy_growth_pct"


def build_uber_model_df() -> pd.DataFrame:
    """
    Construct UBER modeling DataFrame: GOV master + UBER-specific Trends + macro.
    Parallel structure to build_master_df.py but UBER data only.
    """
    gov = pd.read_csv(UBER_GOV_MASTER_PATH, parse_dates=["quarter_end_date"])
    gov = gov.sort_values("quarter_end_date").reset_index(drop=True)

    # Load Trends and extract UBER-specific features
    try:
        trends = pd.read_csv(GOOGLE_TRENDS_PATH, parse_dates=["date"])
        if "Uber Eats" in trends.columns:
            uber_qtly = weekly_to_quarterly(trends, "date", "Uber Eats", 8, 2)
            uber_qtly = uber_qtly.rename(columns={"Uber Eats_mean": "uber_eats_trends_mean"})
            gov = gov.merge(uber_qtly[["quarter_label", "uber_eats_trends_mean"]], on="quarter_label", how="left")
            gov["uber_trends_momentum"] = gov["uber_eats_trends_mean"].pct_change() * 100

        if "three_way_doordash_share" in trends.columns:
            # UBER share = 1 - DASH share (approximate from three-way split)
            uber_share = trends[["date", "three_way_doordash_share"]].copy()
            uber_share["uber_three_way_share"] = 1.0 - uber_share["three_way_doordash_share"]
            uber_qtly2 = weekly_to_quarterly(uber_share, "date", "uber_three_way_share", 8, 2)
            gov = gov.merge(uber_qtly2[["quarter_label", "uber_three_way_share_mean"]], on="quarter_label", how="left")

    except FileNotFoundError:
        print("  Warning: google_trends.csv not found.")

    # Macro features (same as DASH — consumer health is not UBER-specific)
    try:
        fred = pd.read_csv(FRED_MACRO_PATH, parse_dates=["date"])
        if "consumer_health_index" in fred.columns:
            macro_qtly = monthly_to_quarterly(fred, "date", "consumer_health_index", 30)
            gov = gov.merge(macro_qtly[["quarter_label", "consumer_health_index"]], on="quarter_label", how="left")
    except FileNotFoundError:
        print("  Warning: fred_macro.csv not found.")

    # Autoregressive feature
    gov["uber_prior_gov_surprise_pct"] = gov["gov_surprise_pct"].shift(1)

    return gov


def walk_forward_uber_ols(
    df: pd.DataFrame = None,
    features: list[str] = None,
    min_train: int = 8,
) -> tuple[pd.DataFrame, object]:
    """
    Identical walk-forward OLS architecture to model_gov.py, applied to UBER data.
    """
    if df is None:
        df = build_uber_model_df()
    if features is None:
        features = [f for f in UBER_MODEL_FEATURES if f in df.columns]

    avail = df.dropna(subset=[UBER_MODEL_TARGET] + features).reset_index(drop=True)
    val_indices = avail.index[avail.index >= min_train].tolist()

    predictions = []
    models = []

    for idx in val_indices:
        train = avail.iloc[:idx]
        if len(train) < min_train:
            continue
        X_train = sm.add_constant(train[features])
        y_train = train[UBER_MODEL_TARGET]
        X_pred = sm.add_constant(avail.iloc[[idx]][features])
        try:
            model = sm.OLS(y_train, X_train).fit()
            pred = model.predict(X_pred).iloc[0]
            models.append(model)
        except Exception as e:
            continue

        actual = avail.iloc[idx][UBER_MODEL_TARGET]
        predictions.append({
            "quarter_label": avail.iloc[idx]["quarter_label"],
            "actual_yoy_pct": actual,
            "predicted_yoy_pct": pred,
            "error": pred - actual,
        })

    pred_df = pd.DataFrame(predictions)
    final_model = None
    if models:
        X_all = sm.add_constant(avail[features])
        final_model = sm.OLS(avail[UBER_MODEL_TARGET], X_all).fit()

    return pred_df, final_model


def compare_dash_vs_uber_forecast(
    dash_forecast: dict,
    uber_df: pd.DataFrame = None,
) -> dict:
    """
    Compare DASH and UBER Q1 2026 forecasts vs. their respective consensus.
    Returns cross-sectional signal: Long DASH / Short UBER or sector view.
    """
    uber_df = uber_df or build_uber_model_df()
    uber_preds, uber_model = walk_forward_uber_ols(uber_df)

    # Forecast UBER Q1 2026
    q1_row = uber_df[uber_df["quarter_label"] == FORECAST_QUARTER]
    features = [f for f in UBER_MODEL_FEATURES if f in uber_df.columns]
    uber_q1_pred = None
    uber_consensus = None

    if uber_model and not q1_row.empty and not q1_row[features].isna().any().any():
        X_pred = sm.add_constant(q1_row[features])
        uber_q1_pred = uber_model.predict(X_pred).iloc[0]
        uber_consensus = q1_row["consensus_yoy_growth_pct"].iloc[0] if "consensus_yoy_growth_pct" in q1_row else None

    dash_vs_consensus = dash_forecast.get("model_vs_consensus_pp")
    uber_vs_consensus = (
        round(uber_q1_pred - uber_consensus, 2)
        if uber_q1_pred is not None and uber_consensus is not None
        else None
    )

    if dash_vs_consensus is not None and uber_vs_consensus is not None:
        if dash_vs_consensus > 0.5 and uber_vs_consensus <= 0.5:
            signal = "Long DASH / Short UBER (DASH above-consensus, UBER at/below)"
        elif dash_vs_consensus > 0 and uber_vs_consensus > 0:
            signal = "Sector view — both above consensus; weaker single-name conviction"
        else:
            signal = "No cross-sectional signal"
    else:
        signal = "Insufficient data for cross-sectional comparison"

    result = {
        "dash_vs_consensus_pp": dash_vs_consensus,
        "uber_vs_consensus_pp": uber_vs_consensus,
        "cross_sectional_signal": signal,
    }

    print("\nCross-sectional comparison:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


if __name__ == "__main__":
    uber_df = build_uber_model_df()
    preds, model = walk_forward_uber_ols(uber_df)
    preds.to_csv(OUTPUTS_TABLES / "uber_walk_forward_results.csv", index=False)
