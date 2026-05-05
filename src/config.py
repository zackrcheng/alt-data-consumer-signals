"""
config.py — project-wide constants, paths, ticker lists, and hardcoded tables.
All modeling scripts import from here; nothing domain-specific lives elsewhere.
"""

import os
from pathlib import Path

import numpy as np
import random

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # dash_project/
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_FIGURES = ROOT / "outputs" / "figures"
OUTPUTS_TABLES = ROOT / "outputs" / "tables"

# Processed file paths
MASTER_DF_PATH = DATA_PROCESSED / "master_df.csv"
DASH_GOV_MASTER_PATH = DATA_PROCESSED / "dash_gov_master.csv"
UBER_GOV_MASTER_PATH = DATA_PROCESSED / "uber_gov_master.csv"

# Raw file paths
PRICES_DAILY_PATH = DATA_RAW / "prices_daily.csv"
PRICES_QUARTERLY_PATH = DATA_RAW / "prices_quarterly.csv"
GOOGLE_TRENDS_PATH = DATA_RAW / "google_trends.csv"
FRED_MACRO_PATH = DATA_RAW / "fred_macro.csv"
IBES_CONSENSUS_PATH = DATA_RAW / "ibes_consensus.csv"
COMPUSTAT_PATH = DATA_RAW / "compustat_fundamentals.csv"
CRSP_EVENT_STUDY_PATH = DATA_RAW / "crsp_event_study.csv"

# ── Ticker universe ────────────────────────────────────────────────────────────
# MODEL INDEPENDENCE RULES — strictly enforced (see CLAUDE.md §7):
#   DASH GOV model:   DASH data ONLY  → model_gov.py
#   UBER GOV model:   UBER data ONLY  → model_crosssectional.py
#   Panel model:      DASH + CART     → model_transmission.py  (transmission only)
#   UBER pooled with DASH: PROHIBITED

CORE_MODEL = ["DASH", "UBER", "CART"]   # CART = Instacart (IPO Sep 2023)
BENCHMARKS = ["SPY", "XLY"]            # market + consumer discretionary ETF
PEER_COMP = ["ABNB", "LYFT"]           # write-up peer table only; not modeled

ALL_TICKERS = CORE_MODEL + BENCHMARKS

# ── Date constants ─────────────────────────────────────────────────────────────
# DASH IPO: December 2020 → hard floor for GOV target variable
DASH_IPO_DATE = "2020-12-10"
CART_IPO_DATE = "2023-09-19"

# Minimum history for feature characterization (Section 8a/8b)
TRENDS_HISTORY_START = "2018-01-01"
FRED_HISTORY_START = "2010-01-01"
PRICES_HISTORY_START = "2018-01-01"

# Walk-forward validation: minimum 8 quarters training → first validation Q1 2023
WALK_FORWARD_MIN_TRAIN_QUARTERS = 8
WALK_FORWARD_VALIDATION_START = "Q1_2023"

# Q1 2026 is the out-of-sample forecast target (earnings May 6 2026)
FORECAST_QUARTER = "Q1_2026"
Q1_2026_EARNINGS_DATE = "2026-05-06"

# ── Google Trends keywords (US geo only, Section 8a) ─────────────────────────
TRENDS_KEYWORDS = {
    "group1": ["DoorDash", "Uber Eats", "Instacart", "food delivery"],
    "group2": ["DashPass", "Uber One", "grocery delivery"],
    "group3": ["DoorDash app", "order food online", "restaurant delivery"],
}

# Derived feature names (quarterly aggregates)
TRENDS_FEATURE_COLS = [
    "doordash_trends_mean",
    "doordash_trends_momentum",       # QoQ acceleration in DoorDash index
    "doordash_vs_ubereats_mean",      # DoorDash / UberEats ratio
    "doordash_vs_instacart_mean",     # DoorDash / Instacart ratio
    "three_way_share_mean",           # DoorDash / (DoorDash + UberEats + Instacart)
    "dashpass_momentum",              # 4-week rolling mean of DashPass
    "trends_seasonal_adj",            # deseasonalized DoorDash index (fit on 2018+)
]

# Aggregation window: 8 weeks ending 2 weeks before quarter-end (no look-ahead)
TRENDS_WINDOW_WEEKS = 8
TRENDS_LAG_WEEKS = 2

# ── FRED macro series (Section 8b) ────────────────────────────────────────────
FRED_SERIES = {
    "UMCSENT":       "University of Michigan Consumer Sentiment (monthly)",
    "RSAFS":         "Advance Retail Sales - Food Services & Drinking Places (monthly)",
    "CPIUFDNS":      "CPI - Food Away From Home (monthly)",
    "DSPIC96":       "Real Disposable Personal Income (monthly)",
    "UNRATE":        "Unemployment Rate (monthly)",
    "USEPUINDXD":    "Economic Policy Uncertainty Index (daily → resample monthly)",
    # Two-pronged supply-side labor signal for Dasher acquisition cost / EBITDA:
    "JTS4000JOL":    "Job Openings: Trade, Transportation & Utilities (SA, thousands)",
    # ↑ Proxy for gig/delivery labor market tightness. Declining from 2022 peak =
    #   normalizing Dasher acquisition costs = margin tailwind for DASH EBITDA.
    "CES4349200001": "All Employees: Couriers & Messengers (headcount, SA, monthly)",
    # ↑ Narrower scope, direct courier industry employment.
}

MACRO_FEATURE_COLS = [
    "umcsent_qtly",
    "umcsent_mom_pct",
    "rsafs_yoy_pct",
    "cpi_food_mom_pct",
    "consumer_health_index",          # z-score composite of UMCSENT + DSPIC96
    "epu_index",
    "jolts_transport_yoy",            # YoY % change in Trade/Transport/Utilities openings
    "courier_employment_yoy",         # YoY % change in courier & messenger headcount
]

# Most recent monthly value available 30 days before quarter-end (no look-ahead)
MACRO_LAG_DAYS = 30

# ── IBES / consensus feature columns ──────────────────────────────────────────
IBES_FEATURE_COLS = [
    "num_analysts",
    "rev_consensus_est_bn",
    "revision_momentum_pct",          # mean analyst estimate revision in prior 30 days
]

# Consensus snapshot: 60 days before quarter-end
IBES_SNAPSHOT_DAYS_BEFORE_QE = 60

# ── Autoregressive feature columns ────────────────────────────────────────────
AR_FEATURE_COLS = [
    "prior_qtr_gov_surprise_pct",
    "prior_qtr_gov_yoy_pct",
    "prior_qtr_take_rate",
]

# ── OLS model features (starting set, Section 11 Step 2) ──────────────────────
OLS_BASE_FEATURES = [
    "doordash_trends_momentum",
    "three_way_share_mean",
    "consumer_health_index",
    "prior_qtr_gov_surprise_pct",
]

# Primary model target
MODEL_TARGET = "gov_yoy_growth_pct"

# ── Master DataFrame column schema (Section 10) ───────────────────────────────
MASTER_DF_COLS = [
    # Identifiers
    "quarter_label",            # e.g. 'Q1_2025'
    "quarter_end_date",         # e.g. 2025-03-31
    # Target variables
    "gov_actual_bn",            # US Marketplace GOV (billions USD)
    "gov_yoy_growth_pct",       # YoY % change — PRIMARY MODEL TARGET
    "gov_qoq_growth_pct",       # QoQ % change
    "gov_consensus_est_bn",     # consensus estimate (from IR guidance)
    "gov_surprise_pct",         # (actual - consensus) / consensus * 100
    "revenue_actual_bn",        # reported revenue
    "revenue_consensus_est_bn", # IBES consensus 60 days before qtr-end
    "revenue_surprise_pct",     # revenue beat/miss %
    "take_rate",                # revenue / GOV
    "ebitda_actual_bn",         # adjusted EBITDA
    "ebitda_margin_pct",        # EBITDA / revenue * 100
    # Trends features
    *TRENDS_FEATURE_COLS,
    # Macro features
    *MACRO_FEATURE_COLS,
    # IBES features
    *IBES_FEATURE_COLS,
    # Autoregressive features
    *AR_FEATURE_COLS,
]

# ── DASH US Marketplace GOV actuals — hardcoded from IR filings (Section 9) ───
# IMPORTANT: Every figure must be verified against ir.doordash.com before use.
# Units: billions USD. Scope: US Marketplace GOV only (excludes international).
# Q1 2026 intentionally None — this is the out-of-sample forecast target.
GOV_ACTUALS = {
    "Q4_2020": 7.20,   # VERIFY against 10-K
    "Q1_2021": 8.10,   # VERIFY against 10-Q
    "Q2_2021": 10.00,  # VERIFY
    "Q3_2021": 10.40,  # VERIFY
    "Q4_2021": 11.90,  # VERIFY
    "Q1_2022": 12.40,  # VERIFY
    "Q2_2022": 13.10,  # VERIFY
    "Q3_2022": 13.70,  # VERIFY
    "Q4_2022": 14.50,  # VERIFY
    "Q1_2023": 15.10,  # VERIFY
    "Q2_2023": 16.50,  # VERIFY
    "Q3_2023": 17.00,  # VERIFY
    "Q4_2023": 17.60,  # VERIFY
    "Q1_2024": 18.50,  # VERIFY
    "Q2_2024": 19.70,  # VERIFY
    "Q3_2024": 20.00,  # VERIFY
    "Q4_2024": 21.30,  # VERIFY
    "Q1_2025": 21.30,  # VERIFY
    "Q2_2025": 22.90,  # VERIFY
    "Q3_2025": 24.30,  # VERIFY
    "Q4_2025": 25.00,  # VERIFY
    "Q1_2026": None,   # TARGET — earnings May 6 2026
}

# Quarter-end dates for alignment with other time series
QUARTER_END_DATES = {
    "Q4_2020": "2020-12-31",
    "Q1_2021": "2021-03-31",
    "Q2_2021": "2021-06-30",
    "Q3_2021": "2021-09-30",
    "Q4_2021": "2021-12-31",
    "Q1_2022": "2022-03-31",
    "Q2_2022": "2022-06-30",
    "Q3_2022": "2022-09-30",
    "Q4_2022": "2022-12-31",
    "Q1_2023": "2023-03-31",
    "Q2_2023": "2023-06-30",
    "Q3_2023": "2023-09-30",
    "Q4_2023": "2023-12-31",
    "Q1_2024": "2024-03-31",
    "Q2_2024": "2024-06-30",
    "Q3_2024": "2024-09-30",
    "Q4_2024": "2024-12-31",
    "Q1_2025": "2025-03-31",
    "Q2_2025": "2025-06-30",
    "Q3_2025": "2025-09-30",
    "Q4_2025": "2025-12-31",
    "Q1_2026": "2026-03-31",   # forecast quarter
}

# ── DASH earnings event dates (Section 13) ────────────────────────────────────
# Verify all dates before running event study.
EARNINGS_DATES = {
    "Q4_2025": "2026-02-18",
    "Q3_2025": "2025-11-05",
    "Q2_2025": "2025-08-06",
    "Q1_2025": "2025-05-06",
    "Q4_2024": "2025-02-12",
    "Q3_2024": "2024-11-06",
    "Q2_2024": "2024-08-07",
    "Q1_2024": "2024-05-01",
    "Q4_2023": "2024-02-14",
    "Q3_2023": "2023-11-01",
}

# Event study windows (trading days relative to earnings date)
CAR_WINDOWS = {
    "primary": (-1, 2),    # CAR[-1, +2]
    "tight": (0, 1),       # CAR[0, +1]
}

# ── Chart / output standards (Section 14) ─────────────────────────────────────
CHART_DPI = 300
CHART_STYLE = {
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Color palette — muted, colorblind-friendly
COLORS = {
    "dash_primary": "#E8392A",    # DoorDash red (muted)
    "uber": "#1A1A1A",            # Uber black
    "cart": "#F4811F",            # Instacart orange
    "consensus": "#7B8FA1",       # grey for consensus line
    "actual": "#2D6A4F",          # green for actuals
    "forecast": "#E07B39",        # amber for model forecast
    "shading": "#E8F4F8",         # light blue for validation window shading
}
