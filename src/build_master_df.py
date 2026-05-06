"""
build_master_df.py — assemble the quarterly master DataFrame from every input.

One row per DASH fiscal quarter (Q4 2020 → Q1 2026, 22 rows). All features
are constructed with no look-ahead:
  • Trends + appstore: 8-week mean ending 2 weeks before quarter-end.
  • FRED macro:       most recent monthly value 30 days before quarter-end.
  • IBES consensus:   already snapped at IBES_SNAPSHOT_DAYS_BEFORE_QE.
  • Reddit:           same 8-week / 2-week rule as Trends.
  • Weather:          already quarterly.
  • Jobs:             single forward-looking snapshot, attached to Q1 2026.
  • Autoregressive:   shift(1) on the time-sorted GOV master.

Splits the output into three views per project spec §15:
  master_df              — everything
  model_features_df      — MODEL_FEATURE_COLS only (regression input)
  corroborating_df       — CORROBORATING_COLS only (write-up signals)
A runtime assertion enforces that the two feature sets are disjoint.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DASH_GOV_MASTER_PATH, MASTER_DF_PATH,
    COMPUSTAT_PATH, IBES_CONSENSUS_PATH,
    GOOGLE_TRENDS_PATH, APPSTORE_PATH, FRED_MACRO_PATH,
    REDDIT_CONSUMER_PATH, REDDIT_SUPPLY_PATH,
    WEATHER_ANOMALY_PATH, JOB_POSTINGS_PATH,
    QUARTER_END_DATES, FORECAST_QUARTER,
    TRENDS_WINDOW_WEEKS, TRENDS_LAG_WEEKS, MACRO_LAG_DAYS,
    MODEL_FEATURE_COLS, CORROBORATING_COLS, MODEL_TARGET,
)
from src.utils import (
    weekly_to_quarterly, monthly_to_quarterly,
    compute_yoy, compute_qoq, print_pull_summary,
)

# Reddit / appstore reuse the Trends 8-week / 2-week-lag aggregation rule.
APPSTORE_WINDOW_WEEKS = TRENDS_WINDOW_WEEKS
APPSTORE_LAG_WEEKS    = TRENDS_LAG_WEEKS
REDDIT_WINDOW_WEEKS   = TRENDS_WINDOW_WEEKS
REDDIT_LAG_WEEKS      = TRENDS_LAG_WEEKS


# ── small helpers ────────────────────────────────────────────────────────────

def _agg_weekly(df: pd.DataFrame, cols: list[str], date_col: str = "date",
                window: int = TRENDS_WINDOW_WEEKS, lag: int = TRENDS_LAG_WEEKS,
                rename_suffix: str = "_mean") -> pd.DataFrame:
    """Aggregate multiple weekly columns to quarterly via mean over [qe-lag-window, qe-lag)."""
    frames = []
    for c in cols:
        if c not in df.columns:
            continue
        q = weekly_to_quarterly(df, date_col, c, window, lag)
        q = q.rename(columns={f"{c}_mean": f"{c}{rename_suffix}"})
        frames.append(q.set_index("quarter_label")[f"{c}{rename_suffix}"])
    if not frames:
        return pd.DataFrame(columns=["quarter_label"])
    return pd.concat(frames, axis=1).reset_index()


def _agg_weekly_sum(df: pd.DataFrame, col: str, date_col: str = "date",
                    window: int = REDDIT_WINDOW_WEEKS,
                    lag: int = REDDIT_LAG_WEEKS) -> pd.DataFrame:
    """Aggregate a weekly count column to quarterly sum (window-aligned to Trends)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    records = []
    for q_label, qe_str in QUARTER_END_DATES.items():
        qe = pd.Timestamp(qe_str)
        win_end = qe - pd.Timedelta(weeks=lag)
        win_start = win_end - pd.Timedelta(weeks=window)
        mask = (df[date_col] >= win_start) & (df[date_col] < win_end)
        records.append({"quarter_label": q_label, f"{col}_sum": df.loc[mask, col].sum()})
    return pd.DataFrame(records)


# ── feature builders (one per source) ────────────────────────────────────────

def _build_trends_features() -> pd.DataFrame:
    """8-week mean of weekly Trends, plus computed momentum + grubhub YoY."""
    df = pd.read_csv(GOOGLE_TRENDS_PATH, parse_dates=["date"])

    direct_cols = [
        "DoorDash", "Grubhub",
        "doordash_vs_grubhub", "four_way_doordash_share", "three_way_doordash_share",
        "food_delivery_momentum", "dashpass_momentum", "trends_seasonal_adj",
    ]
    agg = _agg_weekly(df, direct_cols)

    # Tidy column names: DoorDash → doordash_trends_mean, Grubhub → grubhub_trends_mean
    agg = agg.rename(columns={
        "DoorDash_mean":                  "doordash_trends_mean",
        "Grubhub_mean":                   "grubhub_trends_mean",
        "doordash_vs_grubhub_mean":       "doordash_vs_grubhub_mean",
        "four_way_doordash_share_mean":   "four_way_doordash_share_mean",
        "three_way_doordash_share_mean":  "three_way_doordash_share_mean",
        "food_delivery_momentum_mean":    "food_delivery_momentum_mean",
        "dashpass_momentum_mean":         "dashpass_momentum_mean",
        "trends_seasonal_adj_mean":       "trends_seasonal_adj_mean",
    })

    # Sort for deterministic momentum
    agg = agg.sort_values("quarter_label",
                          key=lambda s: s.map(QUARTER_END_DATES)).reset_index(drop=True)

    # Computed: doordash_trends_momentum = QoQ % change in doordash quarterly mean
    agg["doordash_trends_momentum"] = compute_qoq(agg["doordash_trends_mean"])

    # Computed: grubhub_yoy_pct (corroborates §14 "Grubhub declining" thesis)
    agg["grubhub_yoy_pct"] = compute_yoy(agg["grubhub_trends_mean"])

    return agg


def _build_appstore_features() -> pd.DataFrame:
    """8-week mean of weekly app store metrics + computed review momentum."""
    df = pd.read_csv(APPSTORE_PATH, parse_dates=["date"])

    direct_cols = [
        "dash_review_count", "dash_engagement_x_sentiment",
        "dash_net_sentiment", "dash_complaint_ratio", "dash_weighted_sentiment",
        "dash_vs_uber_review_ratio", "dash_vs_grubhub_review_ratio",
        "three_way_appstore_share", "five_way_appstore_share",
        "dash_vs_uber_net_sentiment",
        "dash_ios_rank",   # snapshot only — populated for current quarter
    ]
    agg = _agg_weekly(df, direct_cols, window=APPSTORE_WINDOW_WEEKS, lag=APPSTORE_LAG_WEEKS)

    agg = agg.rename(columns={
        "dash_review_count_mean": "dash_review_velocity_mean",  # match §9 naming
    })

    agg = agg.sort_values("quarter_label",
                          key=lambda s: s.map(QUARTER_END_DATES)).reset_index(drop=True)
    agg["dash_review_momentum"] = compute_qoq(agg["dash_review_velocity_mean"])
    return agg


def _build_macro_features() -> pd.DataFrame:
    """Most recent monthly FRED value 30 days before each quarter-end."""
    df = pd.read_csv(FRED_MACRO_PATH, parse_dates=["date"])

    direct_cols = [
        "umcsent", "consumer_health_index", "umcsent_mom_pct",
        "rsafs_yoy_pct", "cpi_food_mom_pct",
        "usepuindxd",
        "jolts_transport_yoy", "courier_employment_yoy",
    ]
    frames = []
    for c in direct_cols:
        if c not in df.columns:
            continue
        q = monthly_to_quarterly(df, "date", c, MACRO_LAG_DAYS)
        frames.append(q.set_index("quarter_label")[c])

    out = pd.concat(frames, axis=1).reset_index()
    return out.rename(columns={
        "umcsent":    "umcsent_qtly",
        "usepuindxd": "epu_index",
    })


def _build_ibes_features() -> pd.DataFrame:
    """DASH-only IBES consensus, already at quarterly grain."""
    df = pd.read_csv(IBES_CONSENSUS_PATH)
    df = df[df["ticker"] == "DASH"].copy()
    keep = [
        "quarter_label", "num_analysts", "revision_momentum_pct",
        "rev_consensus_est_bn", "rev_actual_bn", "rev_surprise_pct",
        "eps_consensus", "eps_actual", "eps_surprise_pct",
    ]
    return df[[c for c in keep if c in df.columns]]


def _build_compustat_features() -> pd.DataFrame:
    """DASH revenue + EBITDA from Compustat (quarterly, already aligned)."""
    df = pd.read_csv(COMPUSTAT_PATH, parse_dates=["quarter_end_date"])
    df = df[df["ticker"] == "DASH"].copy()
    keep = {
        "quarter_label":     "quarter_label",
        "revenue_bn":        "revenue_actual_bn",
        "ebitda_proxy_bn":   "ebitda_actual_bn",
        "ebitda_margin_pct": "ebitda_margin_pct_compustat",
        "gross_margin_pct":  "gross_margin_pct",
        "net_margin_pct":    "net_margin_pct",
    }
    cols = [c for c in keep if c in df.columns]
    return df[cols].rename(columns=keep)


def _build_reddit_features() -> pd.DataFrame:
    """Reddit signals (r/doordash + r/doordash_drivers only), 8wk / 2wk lag."""
    consumer = pd.read_csv(REDDIT_CONSUMER_PATH, parse_dates=["date"])
    supply   = pd.read_csv(REDDIT_SUPPLY_PATH,   parse_dates=["date"])

    dd = consumer[consumer["subreddit"] == "doordash"].copy()
    cons = _agg_weekly(dd, [
        "complaint_ratio", "weighted_sentiment",
        "dash_vs_uber_complaint_ratio", "dash_sentiment_momentum_4wk",
    ], window=REDDIT_WINDOW_WEEKS, lag=REDDIT_LAG_WEEKS)
    cons = cons.rename(columns={
        "complaint_ratio_mean":             "reddit_consumer_complaint_ratio",
        "weighted_sentiment_mean":          "reddit_consumer_weighted_sentiment",
        "dash_vs_uber_complaint_ratio_mean": "reddit_dash_vs_uber_complaint_ratio",
        "dash_sentiment_momentum_4wk_mean":  "reddit_dash_sentiment_momentum_4wk",
    })
    # Signal-availability flag — True iff every week in the window had it True
    sig = _agg_weekly(dd, ["reddit_signal_available"],
                       window=REDDIT_WINDOW_WEEKS, lag=REDDIT_LAG_WEEKS)
    cons["reddit_signal_available"] = (
        sig.get("reddit_signal_available_mean", pd.Series(np.nan)) >= 0.5
    )

    drv = supply[supply["subreddit"] == "doordash_drivers"].copy()
    sup_means = _agg_weekly(drv, [
        "supply_stress_index", "driver_supply_stress_4wk",
    ], window=REDDIT_WINDOW_WEEKS, lag=REDDIT_LAG_WEEKS)
    sup_means = sup_means.rename(columns={
        "supply_stress_index_mean":     "reddit_supply_stress_index",
        "driver_supply_stress_4wk_mean": "reddit_driver_supply_stress_4wk",
    })
    deact = _agg_weekly_sum(drv, "deactivation_mentions",
                             window=REDDIT_WINDOW_WEEKS, lag=REDDIT_LAG_WEEKS)
    peak  = _agg_weekly_sum(drv, "peak_pay_mentions",
                             window=REDDIT_WINDOW_WEEKS, lag=REDDIT_LAG_WEEKS)
    deact = deact.rename(columns={"deactivation_mentions_sum": "reddit_deactivation_mentions"})
    peak  = peak.rename(columns={"peak_pay_mentions_sum":     "reddit_peak_pay_mentions"})

    return (cons.merge(sup_means, on="quarter_label", how="outer")
                .merge(deact, on="quarter_label", how="outer")
                .merge(peak,  on="quarter_label", how="outer"))


def _build_weather_features() -> pd.DataFrame:
    """weather_anomaly.csv is already quarterly."""
    df = pd.read_csv(WEATHER_ANOMALY_PATH)
    keep = [
        "quarter_label",
        "weather_demand_boost_index",
        "weather_demand_boost_index_popwt",
        "extreme_weather_days_composite",
        "cold_snap_days_composite",
    ]
    return df[[c for c in keep if c in df.columns]]


def _build_jobs_snapshot() -> dict[str, float]:
    """
    Forward-looking snapshot (single scrape date). Returns category counts
    suitable for attaching to the FORECAST_QUARTER row only.

    Numerator semantics:
      jobs_dash_merchant_sales_us   = DoorDash merchant-sales roles, US
      jobs_dash_merchant_sales_intl = DoorDash merchant-sales roles, international
      jobs_dash_deliveroo_market    = DoorDash roles in Deliveroo markets (UK/DE/FR/AU)
      jobs_dash_ops_expansion       = DoorDash ops/expansion roles (any geo)
      jobs_dash_dasher_supply       = DoorDash dasher-supply roles (any geo)
      jobs_uber_merchant_sales_us   = Uber merchant-sales US
      jobs_uber_merchant_sales_intl = Uber merchant-sales intl
      jobs_dash_vs_uber_posting_ratio = total DASH / total Uber postings
    """
    df = pd.read_csv(JOB_POSTINGS_PATH)
    if df.empty:
        return {}

    dash = df[df["company"] == "DoorDash"]
    uber = df[df["company"] == "Uber"]

    def _us(d):    return d[d["location_country"] == "United States"]
    def _intl(d):  return d[d["location_country"] != "United States"]

    out = {
        "jobs_dash_merchant_sales_us":    int(_us(dash)["is_merchant_sales"].sum()),
        "jobs_dash_merchant_sales_intl":  int(_intl(dash)["is_merchant_sales"].sum()),
        "jobs_dash_deliveroo_market":     int(dash["is_deliveroo_market"].sum()),
        "jobs_dash_ops_expansion":        int(dash["is_ops_expansion"].sum()),
        "jobs_dash_dasher_supply":        int(dash["is_dasher_supply"].sum()),
        "jobs_uber_merchant_sales_us":    int(_us(uber)["is_merchant_sales"].sum()),
        "jobs_uber_merchant_sales_intl":  int(_intl(uber)["is_merchant_sales"].sum()),
    }
    out["jobs_dash_vs_uber_posting_ratio"] = (
        len(dash) / len(uber) if len(uber) else np.nan
    )
    return out


# ── orchestration ────────────────────────────────────────────────────────────

def build_master_df() -> pd.DataFrame:
    """Merge every source into the quarterly master DataFrame."""

    # 1. GOV master is the spine — defines row index (Q4 2020 → Q1 2026)
    df = pd.read_csv(DASH_GOV_MASTER_PATH, parse_dates=["quarter_end_date"])

    # 2. Compustat: revenue, EBITDA, margins (DASH-only)
    df = df.merge(_build_compustat_features(), on="quarter_label", how="left")

    # Cross-check: revenue implied by FactSet (gov × take_rate) vs Compustat-reported
    df["revenue_implied_bn"] = (
        df["gov_actual_mn"] * df["take_rate_pct"] / 100.0 / 1000.0
    )
    df["ebitda_margin_pct"] = (
        df["ebitda_actual_bn"] / df["revenue_actual_bn"] * 100.0
        if "ebitda_actual_bn" in df.columns and "revenue_actual_bn" in df.columns
        else np.nan
    )

    # 3. Trends (weekly → quarterly)
    df = df.merge(_build_trends_features(), on="quarter_label", how="left")

    # 4. App store (weekly → quarterly)
    df = df.merge(_build_appstore_features(), on="quarter_label", how="left")

    # 5. Macro (monthly → quarterly with 30-day lag)
    df = df.merge(_build_macro_features(), on="quarter_label", how="left")

    # 6. IBES (already quarterly)
    df = df.merge(_build_ibes_features(), on="quarter_label", how="left")

    # 7. Reddit (corroborating, weekly → quarterly with same window as Trends)
    df = df.merge(_build_reddit_features(), on="quarter_label", how="left")

    # 8. Weather (already quarterly, corroborating)
    df = df.merge(_build_weather_features(), on="quarter_label", how="left")

    # 9. Autoregressive features — shift(1) on time-sorted spine
    df = df.sort_values("quarter_end_date").reset_index(drop=True)
    df["prior_qtr_gov_surprise_pct"] = df["gov_surprise_pct"].shift(1)
    df["prior_qtr_gov_yoy_pct"]      = df["gov_yoy_growth_pct"].shift(1)
    df["prior_qtr_orders_yoy_pct"]   = df["orders_yoy_growth_pct"].shift(1)
    df["prior_qtr_take_rate_pct"]    = df["take_rate_pct"].shift(1)

    # 10. Jobs snapshot — single scrape date, attach to FORECAST_QUARTER row only
    snap = _build_jobs_snapshot()
    for col, val in snap.items():
        df[col] = np.where(df["quarter_label"] == FORECAST_QUARTER, val, np.nan)

    # 11. project spec §15 invariant: model features and corroborating must be disjoint
    overlap = set(MODEL_FEATURE_COLS) & set(CORROBORATING_COLS)
    if overlap:
        raise ValueError(f"MODEL_FEATURE_COLS ∩ CORROBORATING_COLS = {overlap}")

    print_pull_summary("Master DataFrame", df, "quarter_end_date")
    return df


def save_master_df() -> None:
    df = build_master_df()
    MASTER_DF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_DF_PATH, index=False)
    print(f"Saved: {MASTER_DF_PATH}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    # Coverage report
    missing = df.drop(columns=["quarter_end_date"]).isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print(f"\nMissing values per column (top 20):")
        print(missing.head(20).to_string())

    # Confirm model features are present and corroborating columns are tagged
    present_model = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    print(f"\nModel features available: {len(present_model)}/{len(MODEL_FEATURE_COLS)}")
    print(f"  present: {present_model}")
    missing_model = set(MODEL_FEATURE_COLS) - set(df.columns)
    if missing_model:
        print(f"  MISSING: {missing_model}")

    present_corr = [c for c in CORROBORATING_COLS if c in df.columns]
    print(f"Corroborating columns available: {len(present_corr)}/{len(CORROBORATING_COLS)}")


if __name__ == "__main__":
    save_master_df()
