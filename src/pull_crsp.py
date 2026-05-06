"""
pull_crsp.py — daily stock returns + market index from WRDS CRSP.

Outputs data/raw/crsp_event_study.csv with columns:
  date, ticker, permno, ret_stock, ret_market_vwretd, abnormal_return,
  prc, vol

ret_stock         daily total return from crsp.dsf
ret_market_vwretd value-weighted CRSP market return from crsp.dsi
abnormal_return   ret_stock − ret_market_vwretd (causal definition; no
                   regression-based abnormal needed for short event windows)

Used by src/event_study.py to compute CAR[-1, +2] and CAR[0, +1] around
DASH/UBER/CART earnings dates and fit β3 (CAR ~ surprise).

Tickers pulled: CORE_MODEL (DASH, UBER, CART). DASH IPO 2020-12-09;
CART IPO 2023-09-19; UBER IPO 2019-05-10. Date range 2018-01-01 onward
(captures DASH from IPO and gives full UBER history).

Mirrors connection logic from pull_wrds_compustat.py / pull_wrds_ibes.py
(.pgpass first, interactive fallback).
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CRSP_EVENT_STUDY_PATH, CORE_MODEL
from src.utils import print_pull_summary

np.random.seed(42)


# ── WRDS connection (consolidated to src/wrds_utils.py) ────────────────────
from src.wrds_utils import get_wrds_connection as _get_wrds_connection


# ── Data pull ────────────────────────────────────────────────────────────────

def _lookup_permnos(db, tickers: list[str]) -> pd.DataFrame:
    """Resolve tickers → CRSP permnos via the CIZ stocknames_v2 table.

    CIZ uses securitybegdt / securityenddt instead of namedt / nameenddt.
    A ticker may map to multiple permnos over time; for DASH/UBER/CART
    (recent IPOs) there's a single permno per ticker.
    """
    ticker_sql = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT DISTINCT permno, ticker, issuernm, securitybegdt, securityenddt
        FROM crsp.stocknames_v2
        WHERE ticker IN ({ticker_sql})
          AND securityenddt >= '2018-01-01'
        ORDER BY ticker, securitybegdt
    """
    df = db.raw_sql(query)
    print(f"  crsp.stocknames_v2 matches:")
    for _, row in df.iterrows():
        print(f"    {row['ticker']:6s}  permno={row['permno']:8.0f}  "
              f"{row['issuernm']:30s}  {row['securitybegdt']} → {row['securityenddt']}")
    return df


def _pull_daily_returns(db, permnos: list[float], start: str = "2018-01-01"
                          ) -> pd.DataFrame:
    """Pull daily total returns from crsp.dsf_v2 (CIZ — coverage through
    2025-12-31). CIZ uses dlycaldt/dlyret/dlyprc/dlyvol column names."""
    permno_sql = ", ".join(str(int(p)) for p in permnos)
    query = f"""
        SELECT dlycaldt AS date, permno, dlyret AS ret,
               dlyprc AS prc, dlyvol AS vol
        FROM crsp.dsf_v2
        WHERE permno IN ({permno_sql})
          AND dlycaldt >= '{start}'
        ORDER BY permno, dlycaldt
    """
    df = db.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  crsp.dsf_v2 (CIZ) returned {len(df):,} stock-day rows  "
          f"(through {df['date'].max().date()})")
    return df


def _pull_market_index(db, start: str = "2018-01-01") -> pd.DataFrame:
    """Pull daily value-weighted market return from crsp.dsi."""
    query = f"""
        SELECT date, vwretd, ewretd, sprtrn
        FROM crsp.dsi
        WHERE date >= '{start}'
        ORDER BY date
    """
    df = db.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  crsp.dsi returned {len(df):,} market-day rows")
    return df


def pull_crsp(tickers: list[str] | None = None, start: str = "2018-01-01"
                ) -> pd.DataFrame:
    """Pull daily CRSP stock + market data, return one merged DataFrame."""
    if tickers is None:
        tickers = list(CORE_MODEL)

    db = _get_wrds_connection()
    try:
        names = _lookup_permnos(db, tickers)
        if names.empty:
            print("  No permnos found — check ticker spelling.")
            return pd.DataFrame()

        # Keep one permno per ticker — most recent securitybegdt wins
        # (handles ticker reuse: e.g. CART = Carolina Trust Bank pre-2019, then Maplebear)
        keep = (names.sort_values(["ticker", "securitybegdt"], ascending=[True, False])
                       .drop_duplicates(subset=["ticker"], keep="first"))
        ticker_to_permno = dict(zip(keep["ticker"], keep["permno"]))
        print(f"\n  Using permnos: {ticker_to_permno}")

        dsf = _pull_daily_returns(db, list(ticker_to_permno.values()), start)
        dsi = _pull_market_index(db, start)
    finally:
        db.close()

    if dsf.empty or dsi.empty:
        print("  Empty CRSP pull — aborting.")
        return pd.DataFrame()

    # Merge ticker symbol back onto dsf via permno
    permno_to_ticker = {v: k for k, v in ticker_to_permno.items()}
    dsf["ticker"] = dsf["permno"].map(permno_to_ticker)

    # Join market return; compute abnormal return
    out = dsf.merge(dsi[["date", "vwretd"]], on="date", how="left")
    out = out.rename(columns={"ret": "ret_stock", "vwretd": "ret_market_vwretd"})
    out["abnormal_return"] = out["ret_stock"] - out["ret_market_vwretd"]

    out = out[["date", "ticker", "permno", "ret_stock", "ret_market_vwretd",
                "abnormal_return", "prc", "vol"]].sort_values(
                ["ticker", "date"]).reset_index(drop=True)

    # ── Per-ticker summary ────────────────────────────────────────────────
    print()
    for tk in tickers:
        sub = out[out["ticker"] == tk]
        if sub.empty:
            print(f"  WARNING: no CRSP rows for {tk}")
            continue
        print(f"  {tk:6s}  permno={int(sub['permno'].iloc[0]):8d}  "
              f"{sub['date'].min().date()} → {sub['date'].max().date()}  "
              f"({len(sub):,} trading days)")

    print_pull_summary("CRSP daily returns (all tickers)", out, "date")
    return out


def save_crsp(tickers: list[str] | None = None, start: str = "2018-01-01") -> None:
    df = pull_crsp(tickers=tickers, start=start)
    if df.empty:
        print("Nothing to save — DataFrame is empty.")
        return
    CRSP_EVENT_STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CRSP_EVENT_STUDY_PATH, index=False)
    print(f"\nSaved {len(df):,} rows → {CRSP_EVENT_STUDY_PATH}")


if __name__ == "__main__":
    save_crsp()
