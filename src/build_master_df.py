"""
build_master_df.py — merge all sources into the single quarterly master DataFrame.

Output: data/processed/master_df.csv
One row per DASH fiscal quarter, Q4 2020 → Q1 2026.
All features pre-aligned with no look-ahead (validated by validate_no_lookahead()).

Pipeline order:
  1. Load GOV master (hardcoded actuals + consensus)
  2. Load Compustat (revenue, EBITDA, take rate = revenue / GOV)
  3. Aggregate Trends features to quarterly (8-week window, 2-week lag)
  4. Aggregate FRED macro features to quarterly (30-day lag)
  5. Load IBES consensus features
  6. Compute autoregressive features (prior-quarter surprise, prior GOV growth)
  7. Merge all and validate no look-ahead
  8. Save to master_df.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    DASH_GOV_MASTER_PATH, COMPUSTAT_PATH, GOOGLE_TRENDS_PATH,
    FRED_MACRO_PATH, IBES_CONSENSUS_PATH, MASTER_DF_PATH,
    TRENDS_WINDOW_WEEKS, TRENDS_LAG_WEEKS, MACRO_LAG_DAYS,
    OLS_BASE_FEATURES, MODEL_TARGET, MASTER_DF_COLS,
)
from src.utils import (
    weekly_to_quarterly, monthly_to_quarterly,
    validate_no_lookahead, print_pull_summary,
)


def build_trends_features(trends_path: Path = GOOGLE_TRENDS_PATH) -> pd.DataFrame:
    """Aggregate weekly Trends to quarterly features (no look-ahead)."""
    try:
        df = pd.read_csv(trends_path, parse_dates=["date"])
    except FileNotFoundError:
        print(f"  Warning: {trends_path} not found. Run pull_trends.py first.")
        return pd.DataFrame()

    feature_map = {
        "DoorDash": "doordash_trends_mean",
        "three_way_doordash_share": "three_way_share_mean",
        "doordash_vs_ubereats": "doordash_vs_ubereats_mean",
        "doordash_vs_instacart": "doordash_vs_instacart_mean",
        "dashpass_momentum": "dashpass_momentum",
    }

    frames = []
    for raw_col, out_col in feature_map.items():
        if raw_col not in df.columns:
            continue
        qtly = weekly_to_quarterly(df, "date", raw_col, TRENDS_WINDOW_WEEKS, TRENDS_LAG_WEEKS)
        qtly = qtly.rename(columns={f"{raw_col}_mean": out_col})
        frames.append(qtly.set_index("quarter_label")[out_col])

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).reset_index()

    # Trends momentum = QoQ change in DoorDash index
    if "doordash_trends_mean" in result.columns:
        result = result.sort_values("quarter_label")
        result["doordash_trends_momentum"] = result["doordash_trends_mean"].pct_change() * 100

    return result


def build_macro_features(fred_path: Path = FRED_MACRO_PATH) -> pd.DataFrame:
    """Aggregate monthly FRED data to quarterly features (no look-ahead)."""
    try:
        df = pd.read_csv(fred_path, parse_dates=["date"])
    except FileNotFoundError:
        print(f"  Warning: {fred_path} not found. Run pull_fred.py first.")
        return pd.DataFrame()

    macro_cols = ["umcsent", "rsafs_yoy_pct", "cpi_food_mom_pct",
                  "consumer_health_index", "umcsent_mom_pct", "usepuindxd"]

    frames = []
    for col in macro_cols:
        if col not in df.columns:
            continue
        qtly = monthly_to_quarterly(df, "date", col, MACRO_LAG_DAYS)
        frames.append(qtly.set_index("quarter_label")[col])

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).reset_index()
    if "umcsent" in result.columns:
        result = result.rename(columns={"umcsent": "umcsent_qtly"})
    if "usepuindxd" in result.columns:
        result = result.rename(columns={"usepuindxd": "epu_index"})

    return result


def build_master_df() -> pd.DataFrame:
    """Merge all sources into the quarterly master DataFrame."""

    # 1. GOV master (foundation)
    gov = pd.read_csv(DASH_GOV_MASTER_PATH, parse_dates=["quarter_end_date"])
    print(f"  GOV master: {len(gov)} rows")

    # 2. Compustat fundamentals (DASH only)
    try:
        comp = pd.read_csv(COMPUSTAT_PATH, parse_dates=["quarter_end_date"])
        comp = comp[comp["ticker"] == "DASH"][
            ["quarter_label", "revenue_bn", "ebitda_proxy_bn"]
        ].rename(columns={"revenue_bn": "revenue_actual_bn", "ebitda_proxy_bn": "ebitda_actual_bn"})
        gov = gov.merge(comp, on="quarter_label", how="left")
    except FileNotFoundError:
        print("  Warning: compustat_fundamentals.csv not found.")

    # Take rate = revenue / GOV
    if "revenue_actual_bn" in gov.columns:
        gov["take_rate"] = gov["revenue_actual_bn"] / gov["gov_actual_bn"]
        gov["ebitda_margin_pct"] = (gov["ebitda_actual_bn"] / gov["revenue_actual_bn"]) * 100

    # 3. Trends features
    trends = build_trends_features()
    if not trends.empty:
        gov = gov.merge(trends, on="quarter_label", how="left")

    # 4. Macro features
    macro = build_macro_features()
    if not macro.empty:
        gov = gov.merge(macro, on="quarter_label", how="left")

    # 5. IBES consensus
    try:
        ibes = pd.read_csv(IBES_CONSENSUS_PATH)
        ibes = ibes[ibes["ticker"] == "DASH"][
            ["quarter_label", "rev_consensus_est_bn", "num_analysts", "revision_momentum_pct"]
        ] if "ticker" in ibes.columns else ibes
        gov = gov.merge(ibes, on="quarter_label", how="left")
    except FileNotFoundError:
        print("  Warning: ibes_consensus.csv not found.")

    # 6. Autoregressive features (shift by 1 quarter — no look-ahead)
    gov = gov.sort_values("quarter_end_date").reset_index(drop=True)
    gov["prior_qtr_gov_surprise_pct"] = gov["gov_surprise_pct"].shift(1)
    gov["prior_qtr_gov_yoy_pct"] = gov["gov_yoy_growth_pct"].shift(1)
    gov["prior_qtr_take_rate"] = gov["take_rate"].shift(1) if "take_rate" in gov.columns else np.nan

    # 7. Validate no look-ahead
    feature_cols = [c for c in OLS_BASE_FEATURES if c in gov.columns]
    validate_no_lookahead(gov, feature_cols, MODEL_TARGET)

    print_pull_summary("Master DataFrame", gov, "quarter_end_date")
    return gov


def save_master_df() -> None:
    df = build_master_df()
    df.to_csv(MASTER_DF_PATH, index=False)
    print(f"Saved: {MASTER_DF_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")


if __name__ == "__main__":
    save_master_df()
