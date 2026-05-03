"""
pull_prices.py — daily and quarterly price data via yfinance.

CRSP is reserved for the earnings event study (event_study.py) only.
yfinance provides adjusted close, volume, returns, and rolling 12-month beta.

Tickers pulled: CORE_MODEL + BENCHMARKS + PEER_COMP (all tickers in one pass).
PEER_COMP (ABNB, LYFT) used only in the write-up peer table — not in GOV models.

Download cascade (in order of preference):
  1. yf.download() bulk — fastest; may be rate-limited after heavy use
  2. Per-ticker Yahoo Finance v8 chart API — different endpoint/bucket
  3. Per-ticker Alpha Vantage TIME_SERIES_DAILY_ADJUSTED — fully independent source
     Requires ALPHA_VANTAGE_API_KEY in .env.  Free tier: 25 calls/day.
"""

import os
import time
import numpy as np
import random
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.config import (
    ALL_TICKERS, PEER_COMP,
    PRICES_HISTORY_START,
    PRICES_DAILY_PATH, PRICES_QUARTERLY_PATH,
    RANDOM_SEED, QUARTER_END_DATES,
)
from src.utils import print_pull_summary

load_dotenv()
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Rolling beta window: 252 trading days ≈ 12 months
BETA_WINDOW_DAYS = 252
BENCHMARK_TICKER = "SPY"


def _fetch_ticker_alpha_vantage(
    ticker: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch full daily adjusted history from Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.
    Requires ALPHA_VANTAGE_API_KEY in .env.  Free tier: 25 calls/day, 5 calls/min.
    Returns DataFrame with [adj_close, volume] indexed by date.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        print(f"    {ticker}: ALPHA_VANTAGE_API_KEY not set — skipping AV fetch.")
        return pd.DataFrame()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "Note" in data:
            print(f"    {ticker}: AV rate limit note: {data['Note'][:80]}")
            return pd.DataFrame()
        if "Information" in data:
            print(f"    {ticker}: AV info: {data['Information'][:80]}")
            return pd.DataFrame()
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            print(f"    {ticker}: AV returned no time series data.")
            return pd.DataFrame()

        rows = []
        for date_str, vals in ts.items():
            rows.append({
                "date": pd.Timestamp(date_str),
                "adj_close": float(vals["5. adjusted close"]),
                "volume": float(vals["6. volume"]),
            })
        df = pd.DataFrame(rows).set_index("date").sort_index()

        # Filter to requested date range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        return df

    except Exception as e:
        print(f"    {ticker}: Alpha Vantage error: {e}")
        return pd.DataFrame()


def _fetch_ticker_v8(
    ticker: str,
    start: str,
    end: str | None,
    max_retries: int = 3,
    base_wait: int = 15,
) -> pd.DataFrame:
    """
    Fetch a single ticker from Yahoo Finance v8 chart API directly.
    Uses query2.finance.yahoo.com which avoids the yfinance bulk-download
    rate limit bucket.  Returns DataFrame with [adj_close, volume] indexed by date.
    """
    end_ts = int(pd.Timestamp(end).timestamp()) if end else int(pd.Timestamp.now().timestamp())
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(pd.Timestamp(start).timestamp()),
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 429:
                wait = base_wait * (2 ** attempt)
                print(f"    {ticker}: 429 rate limit (attempt {attempt + 1}). Waiting {wait}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            result = r.json()["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]
            adjclose_block = result["indicators"].get("adjclose", [{}])[0]
            adj_close = adjclose_block.get("adjclose") or quotes["close"]
            dates = pd.to_datetime(timestamps, unit="s", utc=True).normalize().tz_localize(None)
            df = pd.DataFrame(
                {"adj_close": adj_close, "volume": quotes["volume"]},
                index=dates,
            )
            df.index.name = "date"
            return df.dropna(subset=["adj_close"])
        except Exception as e:
            print(f"    {ticker} error (attempt {attempt + 1}): {e}")
            time.sleep(base_wait)

    print(f"    {ticker}: all {max_retries} attempts failed.")
    return pd.DataFrame()


def _download_with_retry(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """
    Download all tickers.  Tries yf.download() bulk first (fastest when not rate-limited).
    Falls back to per-ticker Yahoo Finance v8 chart API (different endpoint, bypasses
    the bulk-download rate limit bucket).
    """
    # ── Attempt fast bulk download ───────────────────────────────────────────
    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if not raw.empty:
            print("  Bulk yf.download() succeeded.")
            return raw
    except Exception as e:
        print(f"  Bulk download exception: {e}")

    # ── Fallback: per-ticker v8 API ──────────────────────────────────────────
    print("  Bulk download empty/failed — falling back to per-ticker v8 API ...")
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _fetch_ticker_v8(ticker, start=start, end=end)
        if not df.empty:
            frames[ticker] = df
            print(f"    {ticker}: {len(df)} rows  "
                  f"({df.index.min().date()} → {df.index.max().date()})")
        time.sleep(2)  # polite gap between tickers

    if frames:
        print(f"  v8 API succeeded for {len(frames)}/{len(tickers)} tickers.")
        missing = [t for t in tickers if t not in frames]
    else:
        missing = list(tickers)

    # ── Fallback 2: Alpha Vantage for any remaining tickers ─────────────────
    if missing:
        av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if av_key:
            print(f"  Alpha Vantage fallback for: {missing}")
            for ticker in missing:
                df = _fetch_ticker_alpha_vantage(ticker, start=start, end=end)
                if not df.empty:
                    frames[ticker] = df
                    print(f"    {ticker}: {len(df)} rows via Alpha Vantage")
                time.sleep(13)  # AV free tier: 5 calls/min → 12s min between calls
        else:
            print(f"  {len(missing)} tickers missing; ALPHA_VANTAGE_API_KEY not set.")

    if not frames:
        return pd.DataFrame()

    # Assemble MultiIndex DataFrame matching yf.download() output structure
    close_df = pd.DataFrame({t: df["adj_close"] for t, df in frames.items()})
    volume_df = pd.DataFrame({t: df["volume"] for t, df in frames.items()})
    combined = pd.concat({"Close": close_df, "Volume": volume_df}, axis=1)
    combined.columns.names = [None, "ticker"]
    return combined


def pull_daily_prices(
    tickers: list[str] = None,
    start: str = PRICES_HISTORY_START,
    end: str = None,
) -> pd.DataFrame:
    """
    Download daily adjusted close and volume for all tickers.
    Returns long-format DataFrame: [date, ticker, adj_close, volume, daily_return_pct].
    Includes PEER_COMP (ABNB, LYFT) for write-up benchmarking.
    """
    if tickers is None:
        tickers = ALL_TICKERS + PEER_COMP  # deduplicated below

    tickers = list(dict.fromkeys(tickers))  # preserve order, drop dupes

    print(f"Downloading daily prices: {tickers}")
    print(f"  Start: {start}  |  End: {end or 'today'}")

    # Download with exponential backoff — yfinance is rate-limited on bulk requests
    raw = _download_with_retry(tickers, start=start, end=end)

    if raw.empty:
        raise ValueError("yfinance returned empty DataFrame — check tickers and date range.")

    # yfinance returns MultiIndex columns when >1 ticker; flat when exactly 1
    if isinstance(raw.columns, pd.MultiIndex):
        adj_close = raw["Close"].stack().rename("adj_close")
        volume = raw["Volume"].stack().rename("volume")
    else:
        # Single-ticker fallback (shouldn't happen given ALL_TICKERS has 5+)
        adj_close = raw["Close"].rename("adj_close")
        volume = raw["Volume"].rename("volume")
        adj_close.index = pd.MultiIndex.from_arrays(
            [raw.index, [tickers[0]] * len(raw)], names=["date", "ticker"]
        )
        volume.index = adj_close.index

    df = pd.concat([adj_close, volume], axis=1).reset_index()
    df.columns = ["date", "ticker", "adj_close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily return within each ticker (pct change; first row per ticker is NaN)
    df["daily_return_pct"] = df.groupby("ticker")["adj_close"].pct_change() * 100

    print_pull_summary("Daily prices (all tickers)", df, "date")
    _print_ticker_coverage(df)
    return df


def _print_ticker_coverage(df: pd.DataFrame) -> None:
    """Print per-ticker row count and date span for quick inspection."""
    print(f"  {'Ticker':<8} {'Rows':>6}  {'First date':<12}  {'Last date':<12}  {'NaN adj_close':>13}")
    for ticker, grp in df.groupby("ticker"):
        n_nan = grp["adj_close"].isna().sum()
        print(
            f"  {ticker:<8} {len(grp):>6}  "
            f"{str(grp['date'].min().date()):<12}  "
            f"{str(grp['date'].max().date()):<12}  "
            f"{n_nan:>13}"
        )
    print()


def add_rolling_beta(
    daily: pd.DataFrame,
    benchmark: str = BENCHMARK_TICKER,
    window: int = BETA_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Append rolling_beta_12m column: cov(r_ticker, r_benchmark) / var(r_benchmark).
    Requires at least `window` trading days of returns; earlier rows are NaN.
    """
    ret_wide = daily.pivot(index="date", columns="ticker", values="daily_return_pct")

    if benchmark not in ret_wide.columns:
        print(f"  Warning: benchmark '{benchmark}' not found — rolling_beta_12m set to NaN.")
        daily["rolling_beta_12m"] = np.nan
        return daily

    spy_ret = ret_wide[benchmark]
    rolling_spy_var = spy_ret.rolling(window, min_periods=window).var()

    beta_cols: dict[str, pd.Series] = {}
    for ticker in ret_wide.columns:
        rolling_cov = ret_wide[ticker].rolling(window, min_periods=window).cov(spy_ret)
        beta_cols[ticker] = rolling_cov / rolling_spy_var

    beta_wide = pd.DataFrame(beta_cols)
    beta_long = beta_wide.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="rolling_beta_12m"
    )
    daily = daily.merge(beta_long, on=["date", "ticker"], how="left")
    return daily


def build_quarterly_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quarterly returns and end-of-quarter rolling beta from pre-fetched daily data.
    Quarter labels match QUARTER_END_DATES keys (e.g., 'Q1_2025') for master_df alignment.

    Return = (last close in quarter) / (first close in quarter) - 1.
    Rolling beta = last available beta reading within the quarter.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    records = []
    for q_label, qe_str in QUARTER_END_DATES.items():
        qe = pd.Timestamp(qe_str)
        # Quarter start: first day of same quarter
        qs = qe.to_period("Q").start_time.normalize()
        # Inclusive filter: all trading days from qs through qe
        qe_incl = qe + pd.Timedelta(days=1)
        mask = (daily["date"] >= qs) & (daily["date"] < qe_incl)
        quarter_data = daily[mask]

        for ticker, grp in quarter_data.groupby("ticker"):
            grp = grp.sort_values("date")
            if grp.empty or grp["adj_close"].isna().all():
                continue
            valid = grp.dropna(subset=["adj_close"])
            first_close = valid["adj_close"].iloc[0]
            last_close = valid["adj_close"].iloc[-1]
            qtr_return_pct = (last_close / first_close - 1) * 100

            last_beta = (
                grp["rolling_beta_12m"].iloc[-1]
                if "rolling_beta_12m" in grp.columns
                else np.nan
            )

            records.append({
                "quarter_label": q_label,
                "quarter_end_date": qe_str,
                "ticker": ticker,
                "open_price": round(first_close, 4),
                "close_price": round(last_close, 4),
                "qtr_return_pct": round(qtr_return_pct, 4),
                "rolling_beta_12m": round(last_beta, 4) if pd.notna(last_beta) else np.nan,
            })

    qtly = pd.DataFrame(records).sort_values(["quarter_label", "ticker"]).reset_index(drop=True)
    print_pull_summary("Quarterly prices", qtly, "quarter_end_date")
    return qtly


def save_prices() -> None:
    """Pull, enrich, and save prices_daily.csv and prices_quarterly.csv."""
    daily = pull_daily_prices()
    daily = add_rolling_beta(daily)

    daily.to_csv(PRICES_DAILY_PATH, index=False)
    print(f"Saved: {PRICES_DAILY_PATH}  ({len(daily):,} rows)")

    quarterly = build_quarterly_prices(daily)
    quarterly.to_csv(PRICES_QUARTERLY_PATH, index=False)
    print(f"Saved: {PRICES_QUARTERLY_PATH}  ({len(quarterly):,} rows)")


if __name__ == "__main__":
    save_prices()
