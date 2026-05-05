"""
pull_fred.py — FRED macro series via fredapi.

Requires FRED_API_KEY in .env (loaded via python-dotenv).
Pulls back to FRED_HISTORY_START for full business cycle characterization.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.config import FRED_SERIES, FRED_HISTORY_START, FRED_MACRO_PATH
from src.utils import print_pull_summary


def pull_fred_series(
    series_ids: dict[str, str] = FRED_SERIES,
    start: str = FRED_HISTORY_START,
) -> pd.DataFrame:
    """
    Pull all FRED series and return a wide monthly DataFrame.
    Daily series (USEPUINDXD) are resampled to monthly mean.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("fredapi not installed. Run: pip install fredapi")

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not found. Copy .env.template to .env and fill in your key."
        )

    fred = Fred(api_key=api_key)
    frames = {}

    for series_id, description in series_ids.items():
        print(f"  Pulling {series_id}: {description}")
        try:
            s = fred.get_series(series_id, observation_start=start)
            s.index = pd.to_datetime(s.index)
            # Resample daily series to monthly mean
            if "daily" in description.lower() or s.index.freq == "D":
                s = s.resample("MS").mean()
            frames[series_id.lower()] = s
        except Exception as e:
            print(f"  Warning: failed to pull {series_id} — {e}")

    if not frames:
        raise RuntimeError("No FRED series pulled successfully.")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # Derived composite: consumer_health_index = z-score avg of UMCSENT + DSPIC96
    for col in ["umcsent", "dspic96"]:
        if col in df.columns:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()
    z_cols = [c for c in df.columns if c.endswith("_z")]
    if z_cols:
        df["consumer_health_index"] = df[z_cols].mean(axis=1)

    # MoM / YoY % changes
    if "umcsent" in df.columns:
        df["umcsent_mom_pct"] = df["umcsent"].pct_change(fill_method=None) * 100
    if "rsafs" in df.columns:
        df["rsafs_yoy_pct"] = df["rsafs"].pct_change(12, fill_method=None) * 100
    if "cpiufdns" in df.columns:
        df["cpi_food_mom_pct"] = df["cpiufdns"].pct_change(fill_method=None) * 100

    # Supply-side labor signal — YoY % change. Falling = labor market loosening
    # = lower Dasher acquisition cost = EBITDA margin tailwind.
    if "jts4000jol" in df.columns:
        df["jolts_transport_yoy"] = df["jts4000jol"].pct_change(12, fill_method=None) * 100
    if "ces4349200001" in df.columns:
        df["courier_employment_yoy"] = df["ces4349200001"].pct_change(12, fill_method=None) * 100

    print_pull_summary("FRED macro (monthly)", df, "date")
    return df


def save_fred() -> None:
    df = pull_fred_series()
    df.to_csv(FRED_MACRO_PATH, index=False)
    print(f"Saved: {FRED_MACRO_PATH}")


if __name__ == "__main__":
    save_fred()
