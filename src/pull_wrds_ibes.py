"""
pull_wrds_ibes.py — WRDS IBES consensus estimates and actuals.

Pulls for DASH and UBER:
  - Quarterly EPS consensus as of ~60 days before quarter-end
  - Quarterly revenue consensus (SAL preferred; REVPS × shares fallback) where available
  - Actual reported values for surprise computation
  - Estimate revision momentum: mean estimate change in 30 days before snapshot

Actuals sourcing hierarchy (applied separately for EPS and revenue):
  1. IBES actuals (ibes.actu_epsus) — preferred because consensus and actuals come
     from the same database, ensuring an apples-to-apples surprise calculation.
     Compustat may apply restatements retroactively that analysts did not see at
     announcement time, which would distort the beat/miss signal.
  2. Compustat revenue (revenue_bn) — fallback for quarters where IBES actuals are
     unavailable. Flagged in stdout when used.

IBES tracks revenue and EPS — not GOV. GOV consensus is reconstructed
separately in build_gov_table.py from management guidance and IR filings.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Allow running as a top-level script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import (
    IBES_CONSENSUS_PATH,
    IBES_SNAPSHOT_DAYS_BEFORE_QE,
    COMPUSTAT_PATH,
    QUARTER_END_DATES,
)
from src.utils import print_pull_summary

np.random.seed(42)

# ── IBES table names — try in order if one fails ──────────────────────────────
_SUMMARY_TABLES = ["ibes.statsumu_epsus", "ibes.statsum_epsus", "ibes.statsum_xepsus"]
_ACTUALS_TABLE = "ibes.actu_epsus"
_ID_TABLE = "ibes.id"

# Revenue measure codes to try in order
_REV_MEASURES = ["SAL", "SALS", "SALE", "REV", "REVPS"]

# Tolerance band for snapshot selection
_SNAPSHOT_BAND_DAYS = 20   # accept snapshots within ±20 days of target


# ── Connection ─────────────────────────────────────────────────────────────────
# WRDS connection helper consolidated to src/wrds_utils.py — see that module.
from src.wrds_utils import get_wrds_connection as _get_wrds_connection


# ── Identifier mapping ─────────────────────────────────────────────────────────

def _get_oftics(db, tickers: list[str]) -> dict[str, str]:
    """
    Map exchange tickers → IBES oftics via ibes.id.
    Returns {exchange_ticker: oftic}. Falls back to ticker == oftic if id table unavailable.
    """
    ticker_sql = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT DISTINCT ticker, oftic, cname
        FROM {_ID_TABLE}
        WHERE ticker IN ({ticker_sql})
        ORDER BY ticker
    """
    try:
        df = db.raw_sql(query)
        if df.empty:
            print("  WARNING: ibes.id returned no rows; falling back to ticker = oftic.")
            return {t: t for t in tickers}
        # If multiple oftics per ticker, keep most common
        mapping = (
            df.groupby("ticker")["oftic"]
            .agg(lambda x: x.value_counts().index[0])
            .to_dict()
        )
        print(f"  IBES oftic mapping: {mapping}")
        return mapping
    except Exception as e:
        print(f"  WARNING: ibes.id failed ({e}); falling back to ticker = oftic.")
        return {t: t for t in tickers}


# ── Pull summary statistics ────────────────────────────────────────────────────

def _pull_summary(
    db,
    oftics: list[str],
    measure: str,
    summary_table: str,
    start_year: int = 2020,
) -> pd.DataFrame:
    """
    Pull all quarterly summary stats for a given measure from an IBES summary table.
    Returns raw long-format df: oftic, fpedats, statpers, meanest, numest, stdev
    """
    oftic_sql = ", ".join(f"'{o}'" for o in oftics)
    query = f"""
        SELECT oftic, statpers, fpedats, fiscalp, meanest, medest, numest, stdev,
               highest, lowest
        FROM {summary_table}
        WHERE oftic IN ({oftic_sql})
          AND measure = '{measure}'
          AND fiscalp = 'QTR'
          AND fpedats >= '{start_year}-01-01'
        ORDER BY oftic, fpedats, statpers
    """
    try:
        df = db.raw_sql(query)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"  {summary_table} measure={measure} failed: {e}")
        return pd.DataFrame()


def pull_consensus_for_measure(
    db,
    oftics: list[str],
    measure: str,
    start_year: int = 2020,
) -> pd.DataFrame:
    """Try each summary table in order; return first non-empty result."""
    for tbl in _SUMMARY_TABLES:
        df = _pull_summary(db, oftics, measure, tbl, start_year)
        if not df.empty:
            print(f"  Pulled {measure} from {tbl}: {len(df):,} rows")
            return df
    print(f"  WARNING: No IBES data found for measure={measure} in any summary table.")
    return pd.DataFrame()


# ── Pull actuals ───────────────────────────────────────────────────────────────

def pull_actuals(
    db,
    oftics: list[str],
    measure: str,
    start_year: int = 2020,
) -> pd.DataFrame:
    """
    Pull quarterly actuals from ibes.actu_epsus.

    Returns: oftic, pends (period end date), anndats, value (actual value)

    pdicity filter uses IN ('Q', 'QTR') to handle variation across WRDS IBES
    schema versions — some encode quarterly periodicity as 'Q', others as 'QTR'.
    Using 'Q' alone causes the query to silently return 0 rows on schemas that
    store 'QTR', which is the root cause of EPS/revenue actuals appearing as
    missing in the beat/miss table.
    """
    oftic_sql = ", ".join(f"'{o}'" for o in oftics)
    query = f"""
        SELECT oftic, pends, anndats, value, pdicity
        FROM {_ACTUALS_TABLE}
        WHERE oftic IN ({oftic_sql})
          AND measure = '{measure}'
          AND pdicity IN ('Q', 'QTR')
          AND pends >= '{start_year}-01-01'
        ORDER BY oftic, pends
    """
    try:
        df = db.raw_sql(query)
        if df is not None and not df.empty:
            print(f"  IBES actuals measure={measure}: {len(df):,} rows")
            return df
        print(f"  IBES actuals measure={measure}: 0 rows returned")
    except Exception as e:
        print(f"  IBES actuals measure={measure} failed: {e}")
    return pd.DataFrame()


# ── Snapshot selection ─────────────────────────────────────────────────────────

def _snap_to_days_before_qe(
    df: pd.DataFrame,
    days_before: int,
    band: int = _SNAPSHOT_BAND_DAYS,
) -> pd.DataFrame:
    """
    From a long-format summary df (with fpedats and statpers), select the single
    snapshot date closest to `days_before` days before each fiscal period end date.
    """
    if df.empty:
        return df

    df = df.copy()
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["days_before_qe"] = (df["fpedats"] - df["statpers"]).dt.days

    # Keep only snapshots within the tolerance band
    df = df[
        (df["days_before_qe"] >= days_before - band) &
        (df["days_before_qe"] <= days_before + band)
    ]

    if df.empty:
        return df

    # Per (oftic, fpedats): pick the snapshot closest to target days_before
    df["delta"] = (df["days_before_qe"] - days_before).abs()
    idx = df.groupby(["oftic", "fpedats"])["delta"].idxmin()
    return df.loc[idx].drop(columns=["delta"]).reset_index(drop=True)


def _compute_revision_momentum(
    df_all: pd.DataFrame,
    snap_days: int = 60,
    mom_window: int = 30,
) -> pd.DataFrame:
    """
    Revision momentum = change in mean estimate from the snapshot ~(snap_days + mom_window)
    days before QE to the snapshot ~snap_days days before QE.

    Returns df with added column: revision_momentum_pct
    """
    if df_all.empty:
        return df_all

    df_60 = _snap_to_days_before_qe(df_all, snap_days)
    df_90 = _snap_to_days_before_qe(df_all, snap_days + mom_window)

    if df_60.empty or df_90.empty:
        df_60["revision_momentum_pct"] = np.nan
        return df_60

    merged = df_60.merge(
        df_90[["oftic", "fpedats", "meanest"]].rename(columns={"meanest": "meanest_prior"}),
        on=["oftic", "fpedats"],
        how="left",
    )
    merged["revision_momentum_pct"] = np.where(
        merged["meanest_prior"].notna() & (merged["meanest_prior"] != 0),
        (merged["meanest"] - merged["meanest_prior"]) / merged["meanest_prior"].abs() * 100,
        np.nan,
    )
    return merged


# ── Quarter label mapping ──────────────────────────────────────────────────────

def _fpedats_to_quarter_label(fpedats: pd.Series) -> pd.Series:
    """Map fiscal period end date to quarter label (e.g., '2025-03-31' → 'Q1_2025')."""
    q = fpedats.dt.quarter
    y = fpedats.dt.year
    return "Q" + q.astype(str) + "_" + y.astype(str)


# ── Revenue conversion from REVPS → $M ────────────────────────────────────────

def _load_shares_outstanding() -> pd.DataFrame:
    """
    Load shares outstanding from saved Compustat data (cshoq, in millions).
    Returns df with columns: ticker, quarter_label, cshoq_m
    """
    if not COMPUSTAT_PATH.exists():
        print("  NOTE: Compustat not found; REVPS → revenue conversion unavailable.")
        return pd.DataFrame()
    try:
        cs = pd.read_csv(COMPUSTAT_PATH, parse_dates=["quarter_end_date"])
        if "cshoq" not in cs.columns:
            return pd.DataFrame()
        cs["quarter_label"] = "Q" + cs["quarter_end_date"].dt.quarter.astype(str) + "_" + cs["quarter_end_date"].dt.year.astype(str)
        return cs[["ticker", "quarter_label", "cshoq"]].rename(columns={"cshoq": "cshoq_m"})
    except Exception as e:
        print(f"  NOTE: Could not load Compustat for shares ({e}).")
        return pd.DataFrame()


# ── Main pull function ─────────────────────────────────────────────────────────

def pull_ibes(
    tickers: list[str] = ["DASH", "UBER"],
    snapshot_days: int = IBES_SNAPSHOT_DAYS_BEFORE_QE,
    start_year: int = 2020,
) -> pd.DataFrame:
    """
    Pull IBES quarterly EPS + revenue consensus and actuals for given tickers.

    Returns one row per (ticker, quarter_label) with:
      - eps_consensus, eps_actual, eps_surprise_pct
      - rev_consensus_est_bn, rev_actual_bn, rev_surprise_pct
      - num_analysts, revision_momentum_pct (EPS-based)

    Actuals source: IBES preferred (apples-to-apples vs IBES consensus);
    Compustat used as fallback for quarters where IBES actuals are unavailable.

    No look-ahead: consensus snapshots use data available at ~60 days before QE.
    """
    db = _get_wrds_connection()
    oftic_map = _get_oftics(db, tickers)          # {ticker: oftic}
    oftics = list(oftic_map.values())
    reverse_map = {v: k for k, v in oftic_map.items()}

    # ── EPS consensus ──────────────────────────────────────────────────────────
    print("\n--- EPS consensus ---")
    eps_all = pull_consensus_for_measure(db, oftics, "EPS", start_year)
    eps_snap = _compute_revision_momentum(eps_all, snap_days=snapshot_days)

    # ── Revenue consensus (SAL, try fallbacks) ───────────────────────────────
    print("\n--- Revenue consensus ---")
    rev_all = pd.DataFrame()
    rev_measure_used = None
    for m in _REV_MEASURES:
        candidate = pull_consensus_for_measure(db, oftics, m, start_year)
        if not candidate.empty:
            rev_all = candidate
            rev_measure_used = m
            break

    rev_snap = _snap_to_days_before_qe(rev_all, snapshot_days) if not rev_all.empty else pd.DataFrame()
    if rev_measure_used:
        print(f"  Revenue measure used: {rev_measure_used}")
    else:
        print("  Revenue consensus not available in IBES for these tickers.")

    # ── EPS actuals ────────────────────────────────────────────────────────────
    print("\n--- EPS actuals ---")
    eps_act = pull_actuals(db, oftics, "EPS", start_year)

    # ── Revenue actuals (same measure as consensus) ────────────────────────────
    print("\n--- Revenue actuals ---")
    rev_act = pd.DataFrame()
    if rev_measure_used:
        rev_act = pull_actuals(db, oftics, rev_measure_used, start_year)

    db.close()

    # ── Build per-quarter consensus table ─────────────────────────────────────
    if eps_snap.empty:
        print("\nWARNING: No EPS snapshot data returned. Returning empty DataFrame.")
        return pd.DataFrame()

    eps_snap["fpedats"] = pd.to_datetime(eps_snap["fpedats"])
    eps_snap["ticker"] = eps_snap["oftic"].map(reverse_map).fillna(eps_snap["oftic"])
    eps_snap["quarter_label"] = _fpedats_to_quarter_label(eps_snap["fpedats"])
    eps_snap = eps_snap.rename(columns={
        "meanest": "eps_consensus",
        "numest": "num_analysts",
        "stdev": "eps_stdev",
        "days_before_qe": "eps_snap_days_before_qe",
    })
    keep_eps = ["ticker", "quarter_label", "fpedats", "eps_consensus", "num_analysts",
                "eps_stdev", "eps_snap_days_before_qe", "revision_momentum_pct"]
    eps_snap = eps_snap[[c for c in keep_eps if c in eps_snap.columns]]

    # ── Merge EPS actuals ──────────────────────────────────────────────────────
    if not eps_act.empty:
        eps_act["fpedats"] = pd.to_datetime(eps_act["pends"])
        eps_act["ticker"] = eps_act["oftic"].map(reverse_map).fillna(eps_act["oftic"])
        eps_act["quarter_label"] = _fpedats_to_quarter_label(eps_act["fpedats"])
        eps_act = eps_act.rename(columns={"value": "eps_actual"})[
            ["ticker", "quarter_label", "eps_actual", "anndats"]
        ]
        merged = eps_snap.merge(eps_act, on=["ticker", "quarter_label"], how="left")
    else:
        merged = eps_snap.copy()
        merged["eps_actual"] = np.nan
        merged["anndats"] = pd.NaT

    # EPS surprise
    merged["eps_surprise_pct"] = np.where(
        merged["eps_consensus"].notna() & (merged["eps_consensus"] != 0) & merged["eps_actual"].notna(),
        (merged["eps_actual"] - merged["eps_consensus"]) / merged["eps_consensus"].abs() * 100,
        np.nan,
    )

    # ── Merge revenue consensus ────────────────────────────────────────────────
    if not rev_snap.empty:
        rev_snap["fpedats"] = pd.to_datetime(rev_snap["fpedats"])
        rev_snap["ticker"] = rev_snap["oftic"].map(reverse_map).fillna(rev_snap["oftic"])
        rev_snap["quarter_label"] = _fpedats_to_quarter_label(rev_snap["fpedats"])
        rev_snap = rev_snap.rename(columns={"meanest": "rev_consensus_raw"})[
            ["ticker", "quarter_label", "rev_consensus_raw"]
        ]
        merged = merged.merge(rev_snap, on=["ticker", "quarter_label"], how="left")
    else:
        merged["rev_consensus_raw"] = np.nan

    # ── Convert REVPS → revenue $M using shares outstanding ───────────────────
    shares_df = _load_shares_outstanding()
    if not shares_df.empty and rev_measure_used == "REVPS":
        merged = merged.merge(shares_df, on=["ticker", "quarter_label"], how="left")
        # REVPS × shares (M) = total revenue ($M) → ÷ 1000 → $B
        merged["rev_consensus_est_bn"] = (
            merged["rev_consensus_raw"] * merged["cshoq_m"] / 1000.0
        )
    elif rev_measure_used and rev_measure_used != "REVPS":
        # Measure is already total revenue in $M
        merged["rev_consensus_est_bn"] = merged["rev_consensus_raw"] / 1000.0
    else:
        merged["rev_consensus_est_bn"] = np.nan

    # ── Revenue actuals: IBES preferred, Compustat fills remaining gaps ──────────
    # IBES actuals match the consensus database, giving apples-to-apples surprise.
    # Compustat may apply retroactive restatements that analysts did not see at
    # announcement time, which would distort the beat/miss signal if used as primary.
    merged["rev_actual_bn"] = np.nan

    if not rev_act.empty:
        rev_act_proc = rev_act.copy()
        rev_act_proc["fpedats"] = pd.to_datetime(rev_act_proc["pends"])
        rev_act_proc["ticker"] = rev_act_proc["oftic"].map(reverse_map).fillna(rev_act_proc["oftic"])
        rev_act_proc["quarter_label"] = _fpedats_to_quarter_label(rev_act_proc["fpedats"])
        if rev_measure_used == "REVPS" and not shares_df.empty:
            rev_act_proc = rev_act_proc.merge(shares_df, on=["ticker", "quarter_label"], how="left")
            rev_act_proc["rev_actual_bn"] = rev_act_proc["value"] * rev_act_proc["cshoq_m"] / 1000.0
        else:
            rev_act_proc["rev_actual_bn"] = rev_act_proc["value"] / 1000.0
        rev_act_proc = rev_act_proc[["ticker", "quarter_label", "rev_actual_bn"]]
        merged = merged.merge(rev_act_proc, on=["ticker", "quarter_label"], how="left")
        n_filled = merged["rev_actual_bn"].notna().sum()
        print(f"  Revenue actuals: IBES ({rev_measure_used}) — {n_filled} quarters filled")

    # Compustat fills quarters where IBES actuals are unavailable
    missing = merged["rev_actual_bn"].isna()
    if missing.any() and COMPUSTAT_PATH.exists():
        try:
            cs = pd.read_csv(COMPUSTAT_PATH, parse_dates=["quarter_end_date"])
            cs["quarter_label"] = "Q" + cs["quarter_end_date"].dt.quarter.astype(str) + "_" + cs["quarter_end_date"].dt.year.astype(str)
            rev_cs = cs[cs["ticker"].isin(tickers)][["ticker", "quarter_label", "revenue_bn"]].copy()
            merged = merged.merge(
                rev_cs.rename(columns={"revenue_bn": "rev_cs_bn"}),
                on=["ticker", "quarter_label"], how="left",
            )
            fill_mask = missing & merged["rev_cs_bn"].notna()
            if fill_mask.any():
                merged.loc[fill_mask, "rev_actual_bn"] = merged.loc[fill_mask, "rev_cs_bn"]
                print(f"  Revenue actuals: Compustat fallback — {fill_mask.sum()} quarters "
                      f"(may include restatements not visible at announcement)")
            merged = merged.drop(columns=["rev_cs_bn"])
        except Exception as e:
            print(f"  NOTE: Compustat revenue fallback failed ({e}).")

    # Revenue surprise
    merged["rev_surprise_pct"] = np.where(
        merged["rev_consensus_est_bn"].notna() &
        (merged["rev_consensus_est_bn"] != 0) &
        merged["rev_actual_bn"].notna(),
        (merged["rev_actual_bn"] - merged["rev_consensus_est_bn"]) / merged["rev_consensus_est_bn"].abs() * 100,
        np.nan,
    )

    # ── Final cleanup ──────────────────────────────────────────────────────────
    merged = merged.sort_values(["ticker", "quarter_label"]).reset_index(drop=True)

    print_pull_summary("IBES (EPS + Revenue consensus)", merged, "fpedats")
    return merged


# ── Beat/miss table printer ────────────────────────────────────────────────────

def print_beat_miss_table(df: pd.DataFrame, ticker: str = "DASH") -> None:
    """Print a formatted beat/miss verification table for a given ticker."""
    sub = df[df["ticker"] == ticker].copy()
    if sub.empty:
        print(f"\nNo IBES data found for {ticker}.")
        return

    sub = sub.sort_values("quarter_label")

    # Columns to display
    display_cols = {
        "quarter_label": "Quarter",
        "eps_consensus": "EPS_Cons",
        "eps_actual": "EPS_Act",
        "eps_surprise_pct": "EPS_Surp%",
        "rev_consensus_est_bn": "Rev_Cons_Bn",
        "rev_actual_bn": "Rev_Act_Bn",
        "rev_surprise_pct": "Rev_Surp%",
        "num_analysts": "#Analysts",
        "revision_momentum_pct": "RevMom%",
    }
    cols_available = [c for c in display_cols if c in sub.columns]
    display = sub[cols_available].rename(columns=display_cols)

    # Format numerics
    def fmt_float(x, decimals=2):
        if pd.isna(x):
            return "  —  "
        return f"{x:+.{decimals}f}" if "%" in str(x) else f"{x:.{decimals}f}"

    for col in ["EPS_Cons", "EPS_Act"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "  —  ")
    for col in ["EPS_Surp%", "Rev_Surp%", "RevMom%"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "  —  ")
    for col in ["Rev_Cons_Bn", "Rev_Act_Bn"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "  —  ")
    if "#Analysts" in display.columns:
        display["#Analysts"] = display["#Analysts"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "  —  ")

    print(f"\n{'='*80}")
    print(f"  IBES Beat/Miss Table — {ticker}")
    print(f"  EPS surprise: (actual − consensus) / |consensus| × 100")
    print(f"  Rev surprise: IBES actuals preferred vs IBES consensus; Compustat fallback flagged in stdout")
    print(f"  NOTE: IBES tracks revenue/EPS — not GOV. GOV consensus → build_gov_table.py")
    print(f"{'='*80}")
    print(display.to_string(index=False))
    print(f"{'='*80}\n")


# ── Save ───────────────────────────────────────────────────────────────────────

def save_ibes(tickers: list[str] = ["DASH", "UBER"]) -> None:
    df = pull_ibes(tickers=tickers)
    if df.empty:
        print("Nothing to save — IBES returned no data.")
        return
    df.to_csv(IBES_CONSENSUS_PATH, index=False)
    print(f"Saved: {IBES_CONSENSUS_PATH}  ({len(df)} rows)")

    # Print beat/miss table for each ticker
    for tk in tickers:
        print_beat_miss_table(df, ticker=tk)


if __name__ == "__main__":
    save_ibes()
