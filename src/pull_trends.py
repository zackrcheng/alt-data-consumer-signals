"""
pull_trends.py — Google Trends data via pytrends (US geo only).

pytrends caps at 5 years per request; we split into overlapping sub-periods and
rescale each segment to produce a continuous index back to TRENDS_HISTORY_START.
Sleep 2 seconds between requests to avoid 429 rate-limit errors.

Derived features:
  doordash_vs_ubereats        DoorDash / UberEats index
  doordash_vs_instacart       DoorDash / Instacart index
  three_way_doordash_share    DoorDash / (DoorDash + UberEats + Instacart)
  food_delivery_momentum      4-week rolling mean of 'food delivery'
  dashpass_momentum           4-week rolling mean of 'DashPass'
  trends_seasonal_adj         STL-deseasonalized DoorDash index (fit on 2018+ history)

Aggregation (no look-ahead): 8-week window ending 2 weeks before quarter-end.
"""

import time
import random
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from pathlib import Path

from src.config import (
    TRENDS_KEYWORDS, TRENDS_HISTORY_START,
    GOOGLE_TRENDS_PATH, RANDOM_SEED,
)
from src.utils import print_pull_summary

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

REQUEST_SLEEP_SEC = 2
# Overlap between consecutive segments so rescaling has signal to anchor on
OVERLAP_WEEKS = 8


def _build_timeframes(start: str, end: str = None, window_years: int = 4) -> list[str]:
    """
    Split a long date range into ≤5-year sub-periods with OVERLAP_WEEKS overlap.
    Overlap is required so _rescale_to_continuous can anchor consecutive segments.
    Returns list of 'YYYY-MM-DD YYYY-MM-DD' strings.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today()
    overlap = pd.Timedelta(weeks=OVERLAP_WEEKS)
    step = pd.DateOffset(years=window_years)

    frames = []
    seg_start = start_dt
    while seg_start < end_dt:
        seg_end = min(seg_start + step, end_dt)
        frames.append(f"{seg_start.strftime('%Y-%m-%d')} {seg_end.strftime('%Y-%m-%d')}")
        if seg_end >= end_dt:
            break
        # Next segment starts OVERLAP_WEEKS before this one ends
        seg_start = seg_end - overlap
    return frames


def _rescale_to_continuous(segments: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Stitch overlapping segment DataFrames into a single continuous index.
    For each consecutive pair, uses the overlap period mean to compute a
    multiplicative rescaling factor — anchors each new segment to the scale
    of the already-stitched portion.
    """
    if not segments:
        return pd.DataFrame()

    result = segments[0].copy().astype(float)

    for seg in segments[1:]:
        seg = seg.copy().astype(float)
        overlap_idx = result.index.intersection(seg.index)

        if len(overlap_idx) < 2:
            # No usable overlap — just append without rescaling and warn
            print(f"  Warning: < 2 overlap weeks between segments; appending without rescaling.")
            new_rows = seg.loc[~seg.index.isin(result.index)]
            result = pd.concat([result, new_rows]).sort_index()
            continue

        for col in seg.columns:
            if col not in result.columns:
                continue
            base_vals = result.loc[overlap_idx, col]
            seg_vals  = seg.loc[overlap_idx, col]
            # Use non-zero overlap weeks only to avoid divide-by-zero
            mask = (seg_vals > 0) & (base_vals > 0)
            if mask.sum() < 2:
                continue
            scale = base_vals[mask].mean() / seg_vals[mask].mean()
            seg[col] = seg[col] * scale

        new_rows = seg.loc[~seg.index.isin(result.index)]
        result = pd.concat([result, new_rows]).sort_index()

    return result


def _pull_group_with_retry(
    pytrends: TrendReq,
    keywords: list[str],
    timeframe: str,
    geo: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Single segment pull with exponential-backoff retry on 429/500 errors."""
    for attempt in range(max_retries):
        try:
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="")
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                return pd.DataFrame()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            return df
        except Exception as e:
            wait = REQUEST_SLEEP_SEC * (2 ** attempt)
            print(f"    Attempt {attempt+1}/{max_retries} failed for {timeframe}: {e}. "
                  f"Retrying in {wait}s…")
            time.sleep(wait)
    print(f"  ERROR: all {max_retries} attempts failed for {timeframe}. Skipping segment.")
    return pd.DataFrame()


def pull_trends_group(
    keywords: list[str],
    start: str = TRENDS_HISTORY_START,
    geo: str = "US",
) -> pd.DataFrame:
    """
    Pull a single keyword group (≤5 keywords) across the full history,
    stitching overlapping 4-year segments into a continuous index.
    Returns weekly DataFrame indexed by date.
    """
    pytrends = TrendReq(hl="en-US", tz=360)
    timeframes = _build_timeframes(start)
    print(f"  Segments to pull: {len(timeframes)} ({timeframes[0]} … {timeframes[-1]})")
    segments = []

    for i, tf in enumerate(timeframes):
        print(f"    [{i+1}/{len(timeframes)}] {tf} …", end=" ", flush=True)
        df = _pull_group_with_retry(pytrends, keywords[:5], tf, geo)
        if not df.empty:
            segments.append(df)
            print(f"OK ({len(df)} weeks)")
        else:
            print("empty — skipped")
        time.sleep(REQUEST_SLEEP_SEC)

    if not segments:
        print("  ERROR: no data returned for any segment.")
        return pd.DataFrame()

    return _rescale_to_continuous(segments)


def _add_seasonal_adjustment(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trends_seasonal_adj: STL-deseasonalized DoorDash index.
    Fits STL on the weekly 'DoorDash' series using 2018+ history (period=52).
    The adjusted index = trend + residual (seasonal component removed).
    """
    try:
        from statsmodels.tsa.seasonal import STL

        dd_col = "DoorDash"
        if dd_col not in combined.columns:
            print("  Warning: 'DoorDash' column missing — skipping trends_seasonal_adj.")
            return combined

        series = combined.set_index("date")[dd_col].astype(float).sort_index()
        # Drop leading zeros/NaN; STL needs a clean series
        series = series[series > 0].dropna()

        if len(series) < 104:   # need ≥ 2 full years for STL to be meaningful
            print(f"  Warning: only {len(series)} non-zero weeks — skipping STL.")
            combined["trends_seasonal_adj"] = np.nan
            return combined

        # period=52 captures annual seasonality in weekly data
        stl = STL(series, period=52, robust=True)
        fit = stl.fit()
        # Adjusted = trend + residual (seasonal removed, rescaled to original range)
        adjusted = pd.Series(fit.trend + fit.resid, index=series.index, name="trends_seasonal_adj")
        # Clip negatives introduced by STL residual arithmetic
        adjusted = adjusted.clip(lower=0)

        combined = combined.merge(
            adjusted.reset_index().rename(columns={"index": "date"}),
            on="date", how="left"
        )
        print(f"  STL seasonal adjustment fitted on {len(series)} weeks.")

    except Exception as e:
        print(f"  Warning: STL seasonal adjustment failed — {e}. Setting to NaN.")
        combined["trends_seasonal_adj"] = np.nan

    return combined


def pull_all_trends(start: str = TRENDS_HISTORY_START) -> pd.DataFrame:
    """
    Pull all three keyword groups and compute all derived share metrics.
    Returns weekly DataFrame with date column and all derived features.
    """
    print(f"\nPulling Google Trends group 1 ({', '.join(TRENDS_KEYWORDS['group1'])})…")
    g1 = pull_trends_group(TRENDS_KEYWORDS["group1"], start=start)

    print(f"\nPulling Google Trends group 2 ({', '.join(TRENDS_KEYWORDS['group2'])})…")
    g2 = pull_trends_group(TRENDS_KEYWORDS["group2"], start=start)

    print(f"\nPulling Google Trends group 3 ({', '.join(TRENDS_KEYWORDS['group3'])})…")
    g3 = pull_trends_group(TRENDS_KEYWORDS["group3"], start=start)

    # Combine all groups on the weekly date index
    frames = [df for df in [g1, g2, g3] if not df.empty]
    if not frames:
        raise RuntimeError("All three Trends groups returned empty — check network/rate limits.")

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    combined.index.name = "date"
    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)

    # ── Derived share metrics (no look-ahead — computed from same-week values) ──
    dd = "DoorDash"
    ue = "Uber Eats"
    ic = "Instacart"
    gh = "Grubhub"

    if dd in combined.columns and ue in combined.columns:
        combined["doordash_vs_ubereats"] = (
            combined[dd] / combined[ue].replace(0, np.nan)
        )
    if dd in combined.columns and ic in combined.columns:
        combined["doordash_vs_instacart"] = (
            combined[dd] / combined[ic].replace(0, np.nan)
        )
    if dd in combined.columns and gh in combined.columns:
        combined["doordash_vs_grubhub"] = (
            combined[dd] / combined[gh].replace(0, np.nan)
        )
    if all(c in combined.columns for c in [dd, ue, ic]):
        total = combined[[dd, ue, ic]].sum(axis=1).replace(0, np.nan)
        combined["three_way_doordash_share"] = combined[dd] / total
    if all(c in combined.columns for c in [dd, ue, ic, gh]):
        total = combined[[dd, ue, ic, gh]].sum(axis=1).replace(0, np.nan)
        combined["four_way_doordash_share"] = combined[dd] / total

    if "food delivery" in combined.columns:
        combined["food_delivery_momentum"] = combined["food delivery"].rolling(4, min_periods=2).mean()
    if "DashPass" in combined.columns:
        combined["dashpass_momentum"] = combined["DashPass"].rolling(4, min_periods=2).mean()
    if "Grubhub+" in combined.columns:
        combined["grubhub_plus_momentum"] = combined["Grubhub+"].rolling(4, min_periods=2).mean()

    # ── STL seasonal adjustment for DoorDash index ──────────────────────────────
    combined = _add_seasonal_adjustment(combined)

    print_pull_summary("Google Trends (weekly)", combined, "date")
    return combined


def save_trends() -> None:
    df = pull_all_trends()
    df.to_csv(GOOGLE_TRENDS_PATH, index=False)
    print(f"Saved → {GOOGLE_TRENDS_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape:   {df.shape}")


if __name__ == "__main__":
    save_trends()
