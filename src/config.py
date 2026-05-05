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
APPSTORE_PATH = DATA_RAW / "appstore_rankings.csv"
WEATHER_RAW_PATH = DATA_RAW / "weather_raw.csv"
WEATHER_ANOMALY_PATH = DATA_RAW / "weather_anomaly.csv"
REDDIT_CONSUMER_PATH = DATA_RAW / "reddit_consumer.csv"
REDDIT_SUPPLY_PATH = DATA_RAW / "reddit_supply.csv"
JOB_POSTINGS_PATH = DATA_RAW / "job_postings.csv"

# FactSet exports — primary consensus source (Section 7g)
# Layout: skiprows=1, columns:
#   Period | Event Date | After Event | Mean | Surp (%) | Num of Est | Low | High | Guid (Low) | Guid (High) | Price Imp (%)
# Dollar columns (GOV, contribution profit, GB) are millions USD.
FACTSET_DASH_GOV_PATH          = DATA_RAW / "factset_dash_gov.xlsx"             # millions USD (US Marketplace GOV)
FACTSET_DASH_ORDERS_PATH       = DATA_RAW / "factset_dash_orders.xlsx"          # thousands of orders → divide by 1000 for millions
FACTSET_DASH_AOV_PATH          = DATA_RAW / "factset_dash_aov.xlsx"             # USD per order
FACTSET_DASH_TAKERATE_PATH     = DATA_RAW / "factset_dash_takerate.xlsx"        # percent
FACTSET_DASH_CONTRIBUTION_PATH = DATA_RAW / "factset_dash_contribution.xlsx"    # millions USD
FACTSET_UBER_GB_PATH                  = DATA_RAW / "factset_uber_gb.xlsx"                  # TOTAL GB (Mobility+Delivery+Freight), millions USD — cross-check only
FACTSET_UBER_EATS_TAKERATE_PATH       = DATA_RAW / "factset_uber_eats_takerate.xlsx"       # percent
FACTSET_UBER_CONTRIBUTION_MARGIN_PATH = DATA_RAW / "factset_uber_contribution_margin.xlsx" # percent
FACTSET_UBER_MAPC_PATH                = DATA_RAW / "factset_uber_mapc.xlsx"                # millions

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
    "group1": ["DoorDash", "Uber Eats", "Instacart", "Grubhub", "food delivery"],
    "group2": ["DashPass", "Uber One", "Grubhub+", "grocery delivery"],
    "group3": ["DoorDash app", "order food online", "restaurant delivery"],
}

# Derived feature names (quarterly aggregates)
TRENDS_FEATURE_COLS = [
    "doordash_trends_mean",
    "doordash_trends_momentum",       # QoQ acceleration in DoorDash index
    "doordash_vs_ubereats_mean",      # DoorDash / UberEats ratio
    "doordash_vs_instacart_mean",     # DoorDash / Instacart ratio
    "doordash_vs_grubhub_mean",       # DoorDash / Grubhub ratio
    "three_way_share_mean",           # DoorDash / (DoorDash + UberEats + Instacart)
    "four_way_share_mean",            # DoorDash / (DoorDash + UberEats + Instacart + Grubhub)
    "dashpass_momentum",              # 4-week rolling mean of DashPass
    "trends_seasonal_adj",            # deseasonalized DoorDash index (fit on 2018+)
]

# Aggregation window: 8 weeks ending 2 weeks before quarter-end (no look-ahead)
TRENDS_WINDOW_WEEKS = 8
TRENDS_LAG_WEEKS = 2

# ── App store identifiers (Section 8f) ────────────────────────────────────────
# NOT foot traffic — DASH is a delivery app, wrong causal chain (see CLAUDE.md §8f)
# Review velocity = WoW change in cumulative review count = engagement/download proxy
APPS = {
    "DASH":      {"google": "com.dd.doordash",      "ios": 719972451},
    "UBER_EATS": {"google": "com.ubercab.eats",     "ios": 1058959277},
    "INSTACART": {"google": "com.instacart.client", "ios": 545599256},
    "GRUBHUB":   {"google": "com.grubhub.android",  "ios": 302920553},
    "GOPUFF":    {"google": "com.main.gopuff",      "ios": 722804810},
}

APPSTORE_FEATURE_COLS = [
    # Volume / engagement (8-week window mean, 2-week lag)
    "dash_review_velocity_mean",        # mean weekly GP review count (download proxy)
    "dash_review_momentum",             # QoQ acceleration in weekly review count
    # Relative velocity vs. peers
    "dash_vs_uber_review_ratio_mean",
    "dash_vs_grubhub_review_ratio_mean",
    # Volume share
    "three_way_appstore_share_mean",    # DASH / (DASH + UBER + CART)
    "five_way_appstore_share_mean",
    # Sentiment (VADER + star rating)
    "dash_net_sentiment_mean",          # positive_ratio - complaint_ratio
    "dash_complaint_ratio_mean",        # fraction is_complaint reviews
    "dash_weighted_sentiment_mean",     # thumbsUpCount-weighted vader compound
    # Relative sentiment vs. UBER
    "dash_vs_uber_net_sentiment_mean",  # DASH - UBER net_sentiment delta
    # iTunes snapshot (current week only — NaN for historical quarters)
    "dash_ios_rank",                    # Food & Drink rank (lower = better)
]

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
    "prior_qtr_orders_yoy_pct",       # orders is the cleaner volume signal vs GOV
    "prior_qtr_take_rate_pct",
]

# ── OLS candidate features (Section 10, max 5 by VIF) ─────────────────────────
# These are the columns allowed to enter the GOV regression. Anything in
# CORROBORATING_COLS is *forbidden* — model_gov.py asserts the disjoint sets.
MODEL_FEATURE_COLS = [
    "doordash_trends_momentum",
    "dash_engagement_x_sentiment_mean",  # primary appstore signal (CLAUDE.md §9)
    "consumer_health_index",
    "prior_qtr_gov_surprise_pct",
    "revision_momentum_pct",
    "jolts_transport_yoy",                # optional 6th — labor supply control
]

# Corroborating signals — never enter the regression. CLAUDE.md §15 enforces
# the split via this list (build_master_df runs an assertion).
CORROBORATING_COLS = [
    # Weather (already quarterly)
    "weather_demand_boost_index",
    "weather_demand_boost_index_popwt",
    "extreme_weather_days_composite",
    "cold_snap_days_composite",
    # Reddit consumer (r/doordash)
    "reddit_consumer_complaint_ratio",
    "reddit_consumer_weighted_sentiment",
    "reddit_dash_vs_uber_complaint_ratio",
    "reddit_dash_sentiment_momentum_4wk",
    # Reddit supply (r/doordash_drivers)
    "reddit_supply_stress_index",
    "reddit_driver_supply_stress_4wk",
    "reddit_deactivation_mentions",
    "reddit_peak_pay_mentions",
    "reddit_signal_available",
    # Job postings (Q1 2026 snapshot only)
    "jobs_dash_merchant_sales_us",
    "jobs_dash_merchant_sales_intl",
    "jobs_dash_deliveroo_market",
    "jobs_dash_ops_expansion",
    "jobs_dash_dasher_supply",
    "jobs_uber_merchant_sales_us",
    "jobs_uber_merchant_sales_intl",
    "jobs_dash_vs_uber_posting_ratio",
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
    # App store features (Section 8f; available Q1 2025 onward from GP data)
    *APPSTORE_FEATURE_COLS,
    # Macro features
    *MACRO_FEATURE_COLS,
    # IBES features
    *IBES_FEATURE_COLS,
    # Autoregressive features
    *AR_FEATURE_COLS,
]

# ── DASH US Marketplace GOV actuals — hardcoded from IR filings (Section 9) ───
# Verified against IR filings. US Marketplace GOV, millions USD.
GOV_ACTUALS = {
    'Q4_2020': 8_179,   # https://s22.q4cdn.com/280253921/files/doc_financials/2021/q4/DASH-Q4-2021-Shareholder-Letter.pdf
    'Q1_2021': 9_913,   # https://s22.q4cdn.com/280253921/files/doc_financials/2021/q4/DASH-Q4-2021-Shareholder-Letter.pdf
    'Q2_2021': 10_456,  # https://s22.q4cdn.com/280253921/files/doc_financials/2021/q4/DASH-Q4-2021-Shareholder-Letter.pdf
    'Q3_2021': 10_416,  # https://s22.q4cdn.com/280253921/files/doc_financials/2021/q4/DASH-Q4-2021-Shareholder-Letter.pdf
    'Q4_2021': 11_159,  # https://s22.q4cdn.com/280253921/files/doc_financials/2022/q4/DASH_Q4-2022-Shareholder-Letter_FINAL-(1).pdf
    'Q1_2022': 12_353,  # https://s22.q4cdn.com/280253921/files/doc_financials/2022/q4/DASH_Q4-2022-Shareholder-Letter_FINAL-(1).pdf
    'Q2_2022': 13_081,  # https://s22.q4cdn.com/280253921/files/doc_financials/2023/q2/DASH-Q2-23_Earnings-Press-Release.pdf
    'Q3_2022': 13_534,  # https://s22.q4cdn.com/280253921/files/doc_financials/2023/q2/DASH-Q2-23_Earnings-Press-Release.pdf
    'Q4_2022': 14_446,  # https://s22.q4cdn.com/280253921/files/doc_financials/2023/q2/DASH-Q2-23_Earnings-Press-Release.pdf
    'Q1_2023': 15_913,  # https://s22.q4cdn.com/280253921/files/doc_financials/2023/q2/DASH-Q2-23_Earnings-Press-Release.pdf
    'Q2_2023': 16_468,  # https://s22.q4cdn.com/280253921/files/doc_financials/2023/q2/DASH-Q2-23_Earnings-Press-Release.pdf
    'Q3_2023': 16_751,  # https://s22.q4cdn.com/280253921/files/doc_financials/2024/q3/DASH-Q3-2024-Ex-99-1-Press-release.pdf
    'Q4_2023': 17_639,  # https://s22.q4cdn.com/280253921/files/doc_financials/2024/q3/DASH-Q3-2024-Ex-99-1-Press-release.pdf
    'Q1_2024': 19_239,  # https://s22.q4cdn.com/280253921/files/doc_financials/2024/q3/DASH-Q3-2024-Ex-99-1-Press-release.pdf
    'Q2_2024': 19_711,  # https://s22.q4cdn.com/280253921/files/doc_financials/2024/q3/DASH-Q3-2024-Ex-99-1-Press-release.pdf
    'Q3_2024': 20_002,  # https://s22.q4cdn.com/280253921/files/doc_financials/2024/q3/DASH-Q3-2024-Ex-99-1-Press-release.pdf
    'Q4_2024': 21_279,  # https://s22.q4cdn.com/280253921/files/doc_financials/2025/q4/Q4-2025-Earnings-Press-Release.pdf
    'Q1_2025': 23_076,  # https://s22.q4cdn.com/280253921/files/doc_financials/2025/q4/Q4-2025-Earnings-Press-Release.pdf
    'Q2_2025': 24_244,  # https://s22.q4cdn.com/280253921/files/doc_financials/2025/q4/Q4-2025-Earnings-Press-Release.pdf
    'Q3_2025': 25_015,  # https://s22.q4cdn.com/280253921/files/doc_financials/2025/q4/Q4-2025-Earnings-Press-Release.pdf
    'Q4_2025': 29_683,  # https://s22.q4cdn.com/280253921/files/doc_financials/2025/q4/Q4-2025-Earnings-Press-Release.pdf
    'Q1_2026': None,    # TARGET — earnings May 6 2026
}

# ── UBER Delivery Gross Bookings actuals — IR-verified (Section 8) ────────────
# UBER's delivery-segment-only GB. factset_uber_gb.xlsx is TOTAL GB
# (Mobility + Delivery + Freight) so cannot be used for the delivery model.
# Use these for the UBER cross-sectional model; FactSet total GB is for
# consensus/guidance cross-check only.
UBER_GB_DELIVERY_ACTUALS = {
    'Q1_2020':  4_683,
    'Q2_2020':  6_961,
    'Q3_2020':  8_550,
    'Q4_2020': 10_050,
    'Q1_2021': 12_461,
    'Q2_2021': 12_912,
    'Q3_2021': 12_828,
    'Q4_2021': 13_444,
    'Q1_2022': 13_903,
    'Q2_2022': 13_876,
    'Q3_2022': 13_684,
    'Q4_2022': 14_315,
    'Q1_2023': 15_026,
    'Q2_2023': 15_595,
    'Q3_2023': 16_094,
    'Q4_2023': 17_011,
    'Q1_2024': 17_699,
    'Q2_2024': 18_126,
    'Q3_2024': 18_663,
    'Q4_2024': 20_126,
    'Q1_2025': 20_377,
    'Q2_2025': 21_734,
    'Q3_2025': 23_322,
    'Q4_2025': 25_431,
    'Q1_2026': None,   # forecast target
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
# Verified all dates before running event study.
EARNINGS_DATES = {
    "Q4_2025": "2026-02-18",
    "Q3_2025": "2025-11-05",
    "Q2_2025": "2025-08-06",
    "Q1_2025": "2025-05-06",
    "Q4_2024": "2025-02-11",
    "Q3_2024": "2024-10-30",
    "Q2_2024": "2024-08-01",
    "Q1_2024": "2024-05-01",
    "Q4_2023": "2024-02-15",
    "Q3_2023": "2023-11-01",
    "Q2_2023": "2023-08-02",
    "Q1_2023": "2023-05-04",
    "Q4_2022": "2023-02-16",
    "Q3_2022": "2022-11-03",
    "Q2_2022": "2022-08-04",
    "Q1_2022": "2022-05-05",
    "Q4_2021": "2022-02-16",
    "Q3_2021": "2021-11-09",
    "Q2_2021": "2021-08-12",
    "Q1_2021": "2021-05-13",
    "Q4_2020": "2021-02-25",
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
