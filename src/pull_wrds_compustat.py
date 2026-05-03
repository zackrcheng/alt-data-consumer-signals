"""
pull_wrds_compustat.py — quarterly fundamentals from WRDS Compustat (comp.fundq).

Pulls DASH, UBER, CART from comp.fundq for the transmission mechanism regressions,
take rate computation, and peer benchmarking table.

IMPORTANT SCOPE NOTES:
  - GOV is NOT in Compustat. It must come from IR filings (build_gov_table.py).
    take_rate = revenue / GOV is computed in build_master_df.py after merging.
  - UBER data is consolidated (all segments). UBER Delivery GOV is IR-only.
    The cross-sectional model uses UBER IR filings, not Compustat, for GOV.
  - CART (Maplebear Inc, ticker CART) IPO was September 19, 2023.
    Compustat history for CART begins Q3 2023 — flag all CART observations
    as having n ≤ 6 quarters through Q4 2024.
  - oibdpq (Operating Income Before Depreciation) is the Compustat GAAP EBITDA
    proxy. It differs from each company's reported Adjusted EBITDA (non-GAAP).
    Use directionally but annotate this gap in the write-up.
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import COMPUSTAT_PATH, CORE_MODEL
from src.utils import print_pull_summary, compute_yoy, compute_qoq

np.random.seed(42)

# ── Compustat fields ───────────────────────────────────────────────────────────
# Every field is suffixed _q (quarterly) in comp.fundq.
COMPUSTAT_FIELDS = [
    "gvkey", "tic", "conm", "datadate", "fyearq", "fqtr",  # identifiers
    "revtq",    # total revenue ($M) — primary revenue field
    "saleq",    # net sales ($M) — backup if revtq missing
    "oibdpq",   # operating income before D&A — GAAP EBITDA proxy ($M)
    "dpq",      # depreciation & amortization ($M) — for transparency
    "niq",      # net income ($M)
    "cogsq",    # cost of goods sold ($M) — for gross margin
    "xsgaq",    # SG&A expense ($M)
    "xrdq",     # R&D expense ($M)
    "atq",      # total assets ($M)
    "dlttq",    # long-term debt ($M)
    "cheq",     # cash and short-term investments ($M)
    "cshoq",    # common shares outstanding (M shares)
    "prccq",    # stock price at quarter close ($) — backup; use yfinance for prices
]

# Standard comp.fundq filters for non-financial consolidated standardized data
COMPUSTAT_FILTERS = "indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'"


def _read_pgpass_password(hostname: str, username: str) -> str | None:
    """Parse ~/.pgpass for a matching entry and return the password."""
    pgpass = os.path.expanduser("~/.pgpass")
    if not os.path.exists(pgpass):
        return None
    with open(pgpass) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 5:
                continue
            h, _port, _db, u, pw = parts[0], parts[1], parts[2], parts[3], ":".join(parts[4:])
            if (h in (hostname, "*")) and (u in (username, "*")):
                return pw
    return None


def _get_wrds_connection():
    """
    Establish WRDS connection.

    Tries the password cached in ~/.pgpass first (non-interactive). Falls
    through to wrds.Connection(wrds_username=...) if pgpass is absent or
    stale — wrds will prompt interactively, which requires a real TTY.
    Run with `! python -m src.pull_wrds_compustat` in Claude Code to get a TTY.
    """
    try:
        import wrds
    except ImportError:
        raise ImportError("wrds not installed. Run: pip install wrds")

    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise EnvironmentError(
            "WRDS_USERNAME not set. Copy .env.template to .env and add WRDS_USERNAME."
        )

    # Try pgpass first so the script runs non-interactively when credentials are fresh.
    password = _read_pgpass_password("wrds-pgdata.wharton.upenn.edu", username)
    if password:
        try:
            db = wrds.Connection(wrds_username=username, wrds_password=password)
            return db
        except Exception as e:
            print(f"  pgpass credentials failed ({e}). Falling back to interactive login...")

    # Interactive fallback — requires TTY (run with ! prefix in Claude Code).
    return wrds.Connection(wrds_username=username)


def _inspect_schema(db) -> None:
    """Print available Compustat table names if primary query fails — aids debugging."""
    try:
        schema_q = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'comp' LIMIT 20"
        tbls = db.raw_sql(schema_q)
        print(f"  Available comp.* tables: {tbls['table_name'].tolist()}")
    except Exception:
        print("  Could not inspect comp schema.")


def _derive_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived columns from raw Compustat fields.
    All arithmetic uses $M inputs; converts to $B for output columns.
    """
    df = df.copy()

    # Use revtq; fall back to saleq if revtq is entirely missing
    df["revenue_m"] = df["revtq"].fillna(df["saleq"])
    df["revenue_bn"] = df["revenue_m"] / 1000.0

    # GAAP EBITDA proxy (oibdpq). Not equal to Adjusted EBITDA reported by each co.
    df["ebitda_proxy_m"] = df["oibdpq"]
    df["ebitda_proxy_bn"] = df["ebitda_proxy_m"] / 1000.0

    # Margins — guard against zero revenue
    df["ebitda_margin_pct"] = np.where(
        df["revenue_m"].gt(0),
        df["ebitda_proxy_m"] / df["revenue_m"] * 100,
        np.nan,
    )
    df["gross_margin_pct"] = np.where(
        df["revenue_m"].gt(0) & df["cogsq"].notna(),
        (df["revenue_m"] - df["cogsq"]) / df["revenue_m"] * 100,
        np.nan,
    )
    df["net_margin_pct"] = np.where(
        df["revenue_m"].gt(0) & df["niq"].notna(),
        df["niq"] / df["revenue_m"] * 100,
        np.nan,
    )

    # Market cap at quarter-end (backup; prefer yfinance prices for market data)
    df["market_cap_bn"] = np.where(
        df["cshoq"].notna() & df["prccq"].notna(),
        df["cshoq"] * df["prccq"] / 1000.0,
        np.nan,
    )

    # Net debt ($B): long-term debt minus cash
    df["net_debt_bn"] = np.where(
        df["dlttq"].notna() & df["cheq"].notna(),
        (df["dlttq"] - df["cheq"]) / 1000.0,
        np.nan,
    )

    # YoY and QoQ revenue growth per ticker (computed within each company's series)
    growth_records = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("quarter_end_date").copy()
        grp["revenue_yoy_pct"] = compute_yoy(grp["revenue_bn"])
        grp["revenue_qoq_pct"] = compute_qoq(grp["revenue_bn"])
        grp["ebitda_yoy_pct"] = compute_yoy(grp["ebitda_proxy_bn"])
        growth_records.append(grp)

    if growth_records:
        df = pd.concat(growth_records).reset_index(drop=True)
    else:
        df["revenue_yoy_pct"] = np.nan
        df["revenue_qoq_pct"] = np.nan
        df["ebitda_yoy_pct"] = np.nan

    # take_rate placeholder — requires GOV from IR filings, merged in build_master_df.py
    df["take_rate"] = np.nan   # filled later: revenue / GOV (from build_gov_table.py)

    return df


def pull_compustat(
    tickers: list = None,
    start_year: int = 2019,
) -> pd.DataFrame:
    """
    Pull quarterly Compustat fundamentals for DASH, UBER, CART.

    Args:
        tickers: list of tickers; defaults to CORE_MODEL = ['DASH', 'UBER', 'CART']
        start_year: first calendar year to include (default 2019 captures full UBER history)

    Returns:
        Long-format DataFrame, one row per (ticker, quarter), with derived columns.
    """
    if tickers is None:
        tickers = CORE_MODEL

    db = _get_wrds_connection()

    ticker_sql = ", ".join(f"'{t}'" for t in tickers)
    fields_sql = ", ".join(COMPUSTAT_FIELDS)

    query = f"""
        SELECT {fields_sql}
        FROM comp.fundq
        WHERE tic IN ({ticker_sql})
          AND datadate >= '{start_year}-01-01'
          AND {COMPUSTAT_FILTERS}
        ORDER BY tic, datadate
    """

    print(f"Querying comp.fundq for tickers: {tickers} from {start_year}...")

    try:
        df = db.raw_sql(query)
        print(f"  comp.fundq returned {len(df):,} rows.")
    except Exception as e:
        print(f"  comp.fundq query failed: {e}")
        _inspect_schema(db)
        db.close()
        return pd.DataFrame()

    db.close()

    if df.empty:
        print("  Warning: Compustat returned zero rows. Check ticker spelling or date range.")
        return df

    # ── Clean and rename ───────────────────────────────────────────────────────
    df = df.rename(columns={
        "tic": "ticker",
        "conm": "company_name",
        "datadate": "quarter_end_date",
    })
    df["quarter_end_date"] = pd.to_datetime(df["quarter_end_date"])

    # Derive quarter label: Q1_2025 etc.
    df["quarter_label"] = (
        "Q" + df["quarter_end_date"].dt.quarter.astype(str)
        + "_" + df["quarter_end_date"].dt.year.astype(str)
    )

    # ── Computed fundamentals ──────────────────────────────────────────────────
    df = _derive_fundamentals(df)

    # ── Per-ticker summary ─────────────────────────────────────────────────────
    for ticker in tickers:
        sub = df[df["ticker"] == ticker]
        if sub.empty:
            print(f"  WARNING: No rows found for ticker '{ticker}'.")
            if ticker == "CART":
                print("    CART (Maplebear Inc) IPO Sep 2023 — check Compustat coverage.")
        else:
            date_min = sub["quarter_end_date"].min().strftime("%Y-%m-%d")
            date_max = sub["quarter_end_date"].max().strftime("%Y-%m-%d")
            n_qtrs = len(sub)
            missing_rev = sub["revenue_bn"].isna().sum()
            missing_ebitda = sub["ebitda_proxy_bn"].isna().sum()
            print(
                f"  {ticker}: {n_qtrs} quarters | {date_min} → {date_max} | "
                f"revenue missing={missing_rev} | ebitda_proxy missing={missing_ebitda}"
            )

    print_pull_summary("Compustat fundamentals (all tickers)", df, "quarter_end_date")

    # ── Scope warnings ─────────────────────────────────────────────────────────
    print("SCOPE NOTES:")
    print("  - GOV not in Compustat; take_rate column is NaN (filled in build_master_df.py)")
    print("  - UBER revenue is consolidated (all segments); Delivery GOV from IR filings only")
    print("  - CART history begins Q3/Q4 2023 (IPO Sep 2023) — n is very small")
    print("  - ebitda_proxy_bn = oibdpq (GAAP); differs from Adjusted EBITDA in press releases")

    return df


def save_compustat(tickers: list = None, start_year: int = 2019) -> None:
    """Pull and save compustat_fundamentals.csv to data/raw/."""
    df = pull_compustat(tickers=tickers, start_year=start_year)
    if df.empty:
        print("Nothing to save — DataFrame is empty.")
        return
    df.to_csv(COMPUSTAT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows → {COMPUSTAT_PATH}")


if __name__ == "__main__":
    save_compustat()
