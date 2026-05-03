"""
build_gov_table.py — hardcoded GOV actuals and consensus from IR filings.

GOV is not in any database (Compustat, IBES, Bloomberg). Every figure is
manually sourced from ir.doordash.com and hardcoded here.

Scope: US Marketplace GOV only.
  - Q1 2026 has Deliveroo consolidation noise inflating TOTAL reported GOV.
  - Model targets US Marketplace GOV to isolate core business quality.
  - This distinction must be explicit in the write-up.

Run this script to (re)generate:
  data/processed/dash_gov_master.csv
  data/processed/uber_gov_master.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    GOV_ACTUALS, QUARTER_END_DATES,
    DASH_GOV_MASTER_PATH, UBER_GOV_MASTER_PATH,
)
from src.utils import compute_yoy, compute_qoq, print_pull_summary


# ── UBER Delivery GOV actuals — from UBER 10-Qs ───────────────────────────────
# Units: billions USD. Source: UBER Investor Relations / 10-Q filings.
# These are TOTAL Uber Eats (Delivery) bookings, not US-only.
# Used for cross-sectional model in model_crosssectional.py ONLY.
UBER_GOV_ACTUALS = {
    "Q4_2020": 10.05,  # VERIFY against UBER 10-K
    "Q1_2021": 12.46,  # VERIFY
    "Q2_2021": 14.95,  # VERIFY
    "Q3_2021": 15.10,  # VERIFY
    "Q4_2021": 16.58,  # VERIFY
    "Q1_2022": 19.03,  # VERIFY
    "Q2_2022": 17.99,  # VERIFY
    "Q3_2022": 15.00,  # VERIFY
    "Q4_2022": 15.17,  # VERIFY
    "Q1_2023": 15.79,  # VERIFY
    "Q2_2023": 17.26,  # VERIFY
    "Q3_2023": 17.94,  # VERIFY
    "Q4_2023": 18.17,  # VERIFY
    "Q1_2024": 18.67,  # VERIFY
    "Q2_2024": 19.97,  # VERIFY
    "Q3_2024": 20.52,  # VERIFY
    "Q4_2024": 21.25,  # VERIFY
    "Q1_2025": 21.50,  # VERIFY
    "Q2_2025": 22.80,  # VERIFY
    "Q3_2025": 24.10,  # VERIFY
    "Q4_2025": 24.90,  # VERIFY
    "Q1_2026": None,   # forecast target
}

# ── GOV consensus estimates — from management guidance and sell-side commentary ─
# IBES does not track GOV; consensus reconstructed from quarterly earnings calls.
# These are approximate midpoints of guidance ranges. Verify before use.
DASH_GOV_CONSENSUS = {
    "Q1_2024": 18.00,
    "Q2_2024": 19.30,
    "Q3_2024": 19.60,
    "Q4_2024": 20.80,
    "Q1_2025": 20.80,
    "Q2_2025": 22.30,
    "Q3_2025": 23.70,
    "Q4_2025": 24.50,
    "Q1_2026": 25.20,  # consensus entering Q1 2026 earnings — update from sell-side
}

UBER_GOV_CONSENSUS = {
    "Q1_2024": 18.20,
    "Q2_2024": 19.50,
    "Q3_2024": 20.00,
    "Q4_2024": 20.80,
    "Q1_2025": 21.00,
    "Q2_2025": 22.30,
    "Q3_2025": 23.60,
    "Q4_2025": 24.50,
    "Q1_2026": 24.80,  # update before use
}


def build_dash_gov_master() -> pd.DataFrame:
    """
    Construct the DASH GOV master table from hardcoded actuals and consensus.
    Returns DataFrame with GOV actuals, consensus, surprise, YoY/QoQ growth.
    """
    records = []
    quarters = sorted(GOV_ACTUALS.keys(), key=lambda q: QUARTER_END_DATES[q])

    for q in quarters:
        actual = GOV_ACTUALS[q]
        consensus = DASH_GOV_CONSENSUS.get(q, np.nan)
        qe = QUARTER_END_DATES[q]
        surprise = (
            (actual - consensus) / consensus * 100
            if actual is not None and not np.isnan(consensus)
            else np.nan
        )
        records.append({
            "quarter_label": q,
            "quarter_end_date": qe,
            "gov_actual_bn": actual,
            "gov_consensus_est_bn": consensus,
            "gov_surprise_pct": surprise,
        })

    df = pd.DataFrame(records)
    df["quarter_end_date"] = pd.to_datetime(df["quarter_end_date"])
    df = df.sort_values("quarter_end_date").reset_index(drop=True)

    # YoY and QoQ growth (requires actual values; skips NaN rows)
    df["gov_yoy_growth_pct"] = compute_yoy(df["gov_actual_bn"])
    df["gov_qoq_growth_pct"] = compute_qoq(df["gov_actual_bn"])
    df["consensus_yoy_growth_pct"] = compute_yoy(df["gov_consensus_est_bn"])

    print_pull_summary("DASH GOV master", df, "quarter_end_date")
    return df


def build_uber_gov_master() -> pd.DataFrame:
    """Construct UBER Delivery GOV master table (same structure as DASH)."""
    records = []
    quarters = sorted(UBER_GOV_ACTUALS.keys(), key=lambda q: QUARTER_END_DATES[q])

    for q in quarters:
        actual = UBER_GOV_ACTUALS[q]
        consensus = UBER_GOV_CONSENSUS.get(q, np.nan)
        qe = QUARTER_END_DATES[q]
        surprise = (
            (actual - consensus) / consensus * 100
            if actual is not None and not np.isnan(consensus)
            else np.nan
        )
        records.append({
            "quarter_label": q,
            "quarter_end_date": qe,
            "gov_actual_bn": actual,
            "gov_consensus_est_bn": consensus,
            "gov_surprise_pct": surprise,
        })

    df = pd.DataFrame(records)
    df["quarter_end_date"] = pd.to_datetime(df["quarter_end_date"])
    df = df.sort_values("quarter_end_date").reset_index(drop=True)
    df["gov_yoy_growth_pct"] = compute_yoy(df["gov_actual_bn"])
    df["gov_qoq_growth_pct"] = compute_qoq(df["gov_actual_bn"])

    print_pull_summary("UBER GOV master", df, "quarter_end_date")
    return df


def save_gov_tables() -> None:
    dash = build_dash_gov_master()
    dash.to_csv(DASH_GOV_MASTER_PATH, index=False)
    print(f"Saved: {DASH_GOV_MASTER_PATH}")

    uber = build_uber_gov_master()
    uber.to_csv(UBER_GOV_MASTER_PATH, index=False)
    print(f"Saved: {UBER_GOV_MASTER_PATH}")


if __name__ == "__main__":
    save_gov_tables()
