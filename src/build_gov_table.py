"""
build_gov_table.py — assemble per-ticker GOV master tables from FactSet.

FactSet is the primary consensus source (CLAUDE.md §7g). Each xlsx file is
already cleaned and IR-verified by hand; this script does not re-validate
actuals against IR filings.

Outputs (CLAUDE.md §9 schema, units in column names):
  data/processed/dash_gov_master.csv  — DASH US Marketplace GOV + decomposition
  data/processed/uber_gov_master.csv  — UBER Delivery GB (hardcoded actuals)
                                         + total-GB FactSet cross-check
                                         + Eats take rate / contribution / MAPC

UBER note: factset_uber_gb.xlsx is TOTAL Gross Bookings (Mobility + Delivery
+ Freight). Delivery actuals come from the hardcoded UBER_GB_DELIVERY_ACTUALS
table in config.py. No delivery-specific consensus exists in FactSet, so
gb_delivery_surprise_pct is intentionally absent.
"""

import pandas as pd
import numpy as np

from src.config import (
    QUARTER_END_DATES, UBER_GB_DELIVERY_ACTUALS,
    DASH_GOV_MASTER_PATH, UBER_GOV_MASTER_PATH,
    FACTSET_DASH_GOV_PATH, FACTSET_DASH_ORDERS_PATH, FACTSET_DASH_AOV_PATH,
    FACTSET_DASH_TAKERATE_PATH, FACTSET_DASH_CONTRIBUTION_PATH,
    FACTSET_UBER_GB_PATH, FACTSET_UBER_EATS_TAKERATE_PATH,
    FACTSET_UBER_CONTRIBUTION_MARGIN_PATH, FACTSET_UBER_MAPC_PATH,
)
from src.utils import (
    load_factset_table, compute_yoy, compute_qoq, print_pull_summary,
)

# Q2/Q3 2021 GOV figures in FactSet were patched from later DASH shareholder
# letters (CLAUDE.md §7g). Flagged so downstream code can disclose this.
GOV_IR_VERIFIED_QUARTERS = {"Q2_2021", "Q3_2021"}


def _factset_columns(df: pd.DataFrame, prefix: str, qty_suffix: str = "",
                     scale: float = 1.0, keep_event_date: bool = False) -> pd.DataFrame:
    """
    Reshape one FactSet table into project columns. `qty_suffix` (e.g. "_mn",
    "_usd", "_pct") attaches only to the six quantity-bearing columns;
    surprise_pct / num_analysts / event_date keep their native units.
    """
    s = qty_suffix
    out = df[["quarter_label"]].copy()
    out[f"{prefix}_actual{s}"]            = df["actual"] * scale
    out[f"{prefix}_factset_consensus{s}"] = df["consensus_mean"] * scale
    out[f"{prefix}_factset_low{s}"]       = df["low"] * scale
    out[f"{prefix}_factset_high{s}"]      = df["high"] * scale
    out[f"{prefix}_guidance_low{s}"]      = df["guid_low"] * scale
    out[f"{prefix}_guidance_high{s}"]     = df["guid_high"] * scale
    out[f"{prefix}_surprise_pct"]         = df["surprise_pct"]
    out[f"{prefix}_factset_num_analysts"] = df["num_est"]
    if keep_event_date:
        out[f"{prefix}_factset_event_date"] = df["event_date"]
    return out


# ── DASH ─────────────────────────────────────────────────────────────────────

def build_dash_gov_master() -> pd.DataFrame:
    """Build dash_gov_master.csv from the five FactSet DASH exports."""
    gov_raw     = load_factset_table(FACTSET_DASH_GOV_PATH)
    orders_raw  = load_factset_table(FACTSET_DASH_ORDERS_PATH)
    aov_raw     = load_factset_table(FACTSET_DASH_AOV_PATH)
    take_raw    = load_factset_table(FACTSET_DASH_TAKERATE_PATH)
    contrib_raw = load_factset_table(FACTSET_DASH_CONTRIBUTION_PATH)

    gov     = _factset_columns(gov_raw,     "gov",     "_mn",  keep_event_date=True)
    # FactSet orders file is in thousands of orders; divide by 1000 → millions.
    orders  = _factset_columns(orders_raw,  "orders",  "_mn",  scale=1.0 / 1000.0)
    aov     = _factset_columns(aov_raw,     "aov",     "_usd")
    take    = _factset_columns(take_raw,    "take_rate", "_pct")
    contrib = _factset_columns(contrib_raw, "contribution_profit", "_mn")

    df = (gov.merge(orders, on="quarter_label", how="outer")
             .merge(aov, on="quarter_label", how="outer")
             .merge(take, on="quarter_label", how="outer")
             .merge(contrib, on="quarter_label", how="outer"))

    df["quarter_end_date"] = pd.to_datetime(df["quarter_label"].map(QUARTER_END_DATES))
    df = df.sort_values("quarter_end_date").reset_index(drop=True)

    # Derived growth metrics
    df["gov_yoy_growth_pct"]    = compute_yoy(df["gov_actual_mn"])
    df["gov_qoq_growth_pct"]    = compute_qoq(df["gov_actual_mn"])
    df["orders_yoy_growth_pct"] = compute_yoy(df["orders_actual_mn"])
    df["aov_yoy_growth_pct"]    = compute_yoy(df["aov_actual_usd"])

    df["gov_guidance_mid_mn"] = (
        df["gov_guidance_low_mn"] + df["gov_guidance_high_mn"]
    ) / 2

    # Decomposition: orders growth outpacing AOV growth = volume-driven (higher
    # quality / more sustainable per CLAUDE.md §2). NaN-safe.
    df["volume_driven_beat"] = (
        df["orders_yoy_growth_pct"] > df["aov_yoy_growth_pct"]
    ).where(
        df["orders_yoy_growth_pct"].notna() & df["aov_yoy_growth_pct"].notna()
    )

    # contribution_profit / revenue, where revenue = GOV × take_rate / 100
    revenue_mn = df["gov_actual_mn"] * df["take_rate_actual_pct"] / 100.0
    df["contribution_margin_pct"] = df["contribution_profit_actual_mn"] / revenue_mn * 100.0

    df["gov_consensus_source"] = "factset"
    df["gov_ir_verified"] = df["quarter_label"].isin(GOV_IR_VERIFIED_QUARTERS)

    # §9 drops "_actual" on take_rate and contribution_profit for readability.
    df = df.rename(columns={
        "take_rate_actual_pct":            "take_rate_pct",
        "contribution_profit_actual_mn":   "contribution_profit_mn",
    })

    # Final column order (§9 schema)
    cols = [
        "quarter_label", "quarter_end_date",
        # GOV
        "gov_actual_mn", "gov_yoy_growth_pct", "gov_qoq_growth_pct",
        "gov_factset_consensus_mn", "gov_factset_num_analysts",
        "gov_factset_low_mn", "gov_factset_high_mn",
        "gov_guidance_low_mn", "gov_guidance_high_mn", "gov_guidance_mid_mn",
        "gov_surprise_pct", "gov_consensus_source", "gov_ir_verified",
        "gov_factset_event_date",
        # Decomposition
        "orders_actual_mn", "orders_factset_consensus_mn",
        "orders_surprise_pct", "orders_yoy_growth_pct",
        "aov_actual_usd", "aov_yoy_growth_pct",
        "volume_driven_beat",
        # Financials
        "take_rate_pct", "take_rate_factset_consensus_pct",
        "contribution_profit_mn", "contribution_margin_pct",
    ]
    df = df[cols]

    print_pull_summary("DASH GOV master", df, "quarter_end_date")
    return df


# ── UBER ─────────────────────────────────────────────────────────────────────

def build_uber_gov_master() -> pd.DataFrame:
    """
    Build uber_gov_master.csv. Delivery actuals from hardcoded
    UBER_GB_DELIVERY_ACTUALS (FactSet UBER GB is total, not delivery).
    """
    gb_total_raw    = load_factset_table(FACTSET_UBER_GB_PATH)
    eats_take_raw   = load_factset_table(FACTSET_UBER_EATS_TAKERATE_PATH)
    eats_contrib_raw = load_factset_table(FACTSET_UBER_CONTRIBUTION_MARGIN_PATH)
    mapc_raw        = load_factset_table(FACTSET_UBER_MAPC_PATH)

    gb_total     = _factset_columns(gb_total_raw,     "gb_total",                "_mn", keep_event_date=True)
    eats_take    = _factset_columns(eats_take_raw,    "eats_take_rate",          "_pct")
    eats_contrib = _factset_columns(eats_contrib_raw, "eats_contribution_margin", "_pct")
    mapc         = _factset_columns(mapc_raw,         "mapc",                    "_mn")

    delivery = pd.DataFrame([
        {"quarter_label": q, "gb_delivery_actual_mn": v}
        for q, v in UBER_GB_DELIVERY_ACTUALS.items()
    ])

    df = (gb_total.merge(delivery, on="quarter_label", how="outer")
                  .merge(mapc, on="quarter_label", how="outer")
                  .merge(eats_take, on="quarter_label", how="outer")
                  .merge(eats_contrib, on="quarter_label", how="outer"))

    # Restrict to the DASH-aligned timeline (Q4 2020+). UBER_GB_DELIVERY_ACTUALS
    # includes Q1–Q3 2020 for reference but those quarters predate DASH's IPO
    # and aren't used downstream.
    df = df[df["quarter_label"].isin(QUARTER_END_DATES)].copy()
    df["quarter_end_date"] = pd.to_datetime(df["quarter_label"].map(QUARTER_END_DATES))
    df = df.sort_values("quarter_end_date").reset_index(drop=True)

    df["gb_delivery_yoy_growth_pct"] = compute_yoy(df["gb_delivery_actual_mn"])
    df["gb_delivery_qoq_growth_pct"] = compute_qoq(df["gb_delivery_actual_mn"])
    df["gb_total_guidance_mid_mn"] = (
        df["gb_total_guidance_low_mn"] + df["gb_total_guidance_high_mn"]
    ) / 2

    # §9 drops "_actual" on the rate / margin metrics for readability.
    df = df.rename(columns={
        "eats_take_rate_actual_pct":           "eats_take_rate_pct",
        "eats_contribution_margin_actual_pct": "eats_contribution_margin_pct",
    })

    cols = [
        "quarter_label", "quarter_end_date",
        # Delivery (model target — hardcoded actuals)
        "gb_delivery_actual_mn", "gb_delivery_yoy_growth_pct", "gb_delivery_qoq_growth_pct",
        # Total GB (FactSet — cross-check + consensus/guidance only)
        "gb_total_actual_mn", "gb_total_factset_consensus_mn",
        "gb_total_surprise_pct", "gb_total_factset_num_analysts",
        "gb_total_factset_low_mn", "gb_total_factset_high_mn",
        "gb_total_guidance_low_mn", "gb_total_guidance_high_mn", "gb_total_guidance_mid_mn",
        "gb_total_factset_event_date",
        # Engagement
        "mapc_actual_mn", "mapc_factset_consensus_mn",
        # Eats financials
        "eats_take_rate_pct", "eats_take_rate_factset_consensus_pct",
        "eats_contribution_margin_pct", "eats_contribution_margin_factset_consensus_pct",
    ]
    df = df[cols]

    print_pull_summary("UBER GOV master", df, "quarter_end_date")
    return df


def save_gov_tables() -> None:
    DASH_GOV_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)

    dash = build_dash_gov_master()
    dash.to_csv(DASH_GOV_MASTER_PATH, index=False)
    print(f"Saved: {DASH_GOV_MASTER_PATH}")

    uber = build_uber_gov_master()
    uber.to_csv(UBER_GOV_MASTER_PATH, index=False)
    print(f"Saved: {UBER_GOV_MASTER_PATH}")


if __name__ == "__main__":
    save_gov_tables()
