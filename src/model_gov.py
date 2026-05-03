"""
model_gov.py — DASH US Marketplace GOV forecast model.

TRAINED ON DASH DATA ONLY. Never mix UBER or CART data here.
Model architecture: OLS primary, Ridge secondary (robustness only).
Validation: expanding window walk-forward, minimum 8 quarters training.

See CLAUDE.md §11 for full modeling architecture spec.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from src.config import (
    MASTER_DF_PATH, OLS_BASE_FEATURES, MODEL_TARGET,
    WALK_FORWARD_MIN_TRAIN_QUARTERS, WALK_FORWARD_VALIDATION_START,
    FORECAST_QUARTER, OUTPUTS_TABLES, RANDOM_SEED,
)
from src.utils import validate_no_lookahead


BASELINES = ["naive", "consensus", "trailing_4q_avg"]


def load_model_data(path: Path = MASTER_DF_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["quarter_end_date"])
    df = df.sort_values("quarter_end_date").reset_index(drop=True)
    return df


def compute_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline 1: Naive — prior quarter GOV (zero growth assumption)
    Baseline 2: Consensus — GOV consensus estimate
    Baseline 3: Trailing 4-quarter average YoY growth
    """
    df = df.copy()
    # Naive: prior quarter actual (implies same absolute GOV → 0 QoQ growth)
    df["pred_naive"] = df["gov_actual_bn"].shift(1)
    df["pred_naive_yoy"] = df["gov_yoy_growth_pct"].shift(1)   # carry forward last YoY

    # Consensus
    df["pred_consensus_yoy"] = df["consensus_yoy_growth_pct"] if "consensus_yoy_growth_pct" in df.columns else np.nan

    # Trailing 4Q avg YoY
    df["pred_trailing4q_yoy"] = df["gov_yoy_growth_pct"].shift(1).rolling(4).mean()

    return df


def walk_forward_ols(
    df: pd.DataFrame,
    features: list[str] = OLS_BASE_FEATURES,
    target: str = MODEL_TARGET,
    min_train: int = WALK_FORWARD_MIN_TRAIN_QUARTERS,
    validation_start: str = WALK_FORWARD_VALIDATION_START,
) -> tuple[pd.DataFrame, list]:
    """
    Expanding window walk-forward OLS validation.

    For each quarter T in the validation window:
      - Train on all quarters through T-1 (minimum min_train quarters)
      - Predict T
      - Record predicted vs. actual, direction hit/miss

    Returns:
      - predictions DataFrame
      - list of fitted OLS model objects (one per validation step)
    """
    validate_no_lookahead(df, features, target)

    # Filter to rows where target and all features are available
    avail = df.dropna(subset=[target] + features).copy()
    avail = avail.reset_index(drop=True)

    # Find validation start index
    val_mask = avail["quarter_label"] >= validation_start
    val_indices = avail.index[val_mask].tolist()

    if len(val_indices) == 0:
        print(f"  No validation quarters found from {validation_start}.")
        return pd.DataFrame(), []

    predictions = []
    models = []

    for idx in val_indices:
        train = avail.iloc[:idx]
        if len(train) < min_train:
            continue

        X_train = sm.add_constant(train[features])
        y_train = train[target]
        X_pred = sm.add_constant(avail.iloc[[idx]][features])

        try:
            model = sm.OLS(y_train, X_train).fit()
            pred = model.predict(X_pred).iloc[0]
            models.append(model)
        except Exception as e:
            print(f"  OLS failed for {avail.iloc[idx]['quarter_label']}: {e}")
            continue

        actual = avail.iloc[idx][target]
        direction_correct = int(np.sign(pred - actual) == 0 or
                                (pred > 0) == (actual > 0))   # same sign

        predictions.append({
            "quarter_label": avail.iloc[idx]["quarter_label"],
            "quarter_end_date": avail.iloc[idx]["quarter_end_date"],
            "actual_yoy_pct": actual,
            "predicted_yoy_pct": pred,
            "error": pred - actual,
            "abs_error": abs(pred - actual),
            "direction_correct": direction_correct,
            "n_train": len(train),
        })

    pred_df = pd.DataFrame(predictions)
    if not pred_df.empty:
        rmse = np.sqrt((pred_df["abs_error"] ** 2).mean())
        dir_acc = pred_df["direction_correct"].mean()
        print(f"\nWalk-forward results ({len(pred_df)} validation quarters):")
        print(f"  RMSE: {rmse:.2f}pp  |  Directional accuracy: {dir_acc:.0%}")

    return pred_df, models


def fit_final_ols(
    df: pd.DataFrame,
    features: list[str] = OLS_BASE_FEATURES,
    target: str = MODEL_TARGET,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS on all available data (for final Q1 2026 forecast)."""
    avail = df.dropna(subset=[target] + features)
    X = sm.add_constant(avail[features])
    y = avail[target]
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


def forecast_q1_2026(
    df: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    features: list[str] = OLS_BASE_FEATURES,
) -> dict:
    """
    Generate Q1 2026 forecast using the final fitted model.
    Requires Q1 2026 features to be present in df (Trends/macro available before Mar 31 2026).

    Returns forecast dict matching the output format in CLAUDE.md §11.
    """
    q1_row = df[df["quarter_label"] == FORECAST_QUARTER]
    if q1_row.empty:
        print(f"  Warning: {FORECAST_QUARTER} not found in master_df.")
        return {}

    # Check feature availability
    missing = [f for f in features if q1_row[f].isna().any()]
    if missing:
        print(f"  Warning: missing features for {FORECAST_QUARTER}: {missing}")

    X_pred = sm.add_constant(q1_row[features])
    pred = model.predict(X_pred).iloc[0]

    # 80% CI from model prediction interval
    pred_result = model.get_prediction(X_pred)
    ci = pred_result.conf_int(alpha=0.20).iloc[0]

    consensus_yoy = q1_row["consensus_yoy_growth_pct"].iloc[0] if "consensus_yoy_growth_pct" in q1_row else np.nan
    vs_consensus = pred - consensus_yoy if not np.isnan(consensus_yoy) else np.nan
    magnitude = (
        "Small (<1pp)" if abs(vs_consensus) < 1
        else "Medium (1–3pp)" if abs(vs_consensus) <= 3
        else "Large (>3pp)"
    ) if not np.isnan(vs_consensus) else "Unknown"

    result = {
        "quarter": FORECAST_QUARTER,
        "predicted_gov_yoy_pct": round(pred, 2),
        "consensus_gov_yoy_pct": round(consensus_yoy, 2) if not np.isnan(consensus_yoy) else None,
        "model_vs_consensus_pp": round(vs_consensus, 2) if not np.isnan(vs_consensus) else None,
        "direction": "Beat" if vs_consensus > 0 else "Miss",
        "magnitude": magnitude,
        "ci_80_low": round(ci[0], 2),
        "ci_80_high": round(ci[1], 2),
    }

    print("\n" + "="*50)
    print(f"Q1 2026 GOV FORECAST")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("="*50)
    return result


if __name__ == "__main__":
    df = load_model_data()
    df = compute_baselines(df)
    pred_df, models = walk_forward_ols(df)
    if models:
        final_model = fit_final_ols(df)
        forecast = forecast_q1_2026(df, final_model)
        pred_df.to_csv(OUTPUTS_TABLES / "walk_forward_results.csv", index=False)
