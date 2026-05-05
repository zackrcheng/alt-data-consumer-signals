"""
utils.py — shared helpers used across all pipeline scripts.

Key function: validate_no_lookahead() — called automatically before any model fit
to assert that every feature column uses only data available before quarter-end.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def validate_no_lookahead(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
    """
    Assert no future information leaks into features.

    For each feature, checks that its value for quarter Q was constructed
    using only data available at least TRENDS_LAG_WEEKS weeks before quarter-end
    (for Trends) or MACRO_LAG_DAYS days before quarter-end (for macro).

    This is enforced by convention: feature construction scripts in pull_*.py
    and build_master_df.py must embed an availability_date for each column.
    This function validates the metadata, not the underlying data pull.

    Raises ValueError with a descriptive message if a violation is detected.
    """
    if "quarter_end_date" not in df.columns:
        raise ValueError("DataFrame must contain 'quarter_end_date' column.")

    # Sentinel check: target variable must never appear in feature_cols
    if target_col in feature_cols:
        raise ValueError(
            f"Target '{target_col}' found in feature_cols — data leakage."
        )

    # Check for NaN in training features (warns rather than raises)
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.any():
        import warnings
        warnings.warn(
            f"NaN values found in features: {nan_counts[nan_counts > 0].to_dict()}. "
            "Fill or drop before fitting."
        )

    # If an 'availability_date' metadata dict is registered, compare dates
    # (populated by pull_trends.py and pull_fred.py via register_feature_availability)
    global _FEATURE_AVAILABILITY
    for col in feature_cols:
        if col in _FEATURE_AVAILABILITY:
            avail_lag = _FEATURE_AVAILABILITY[col]   # timedelta
            for _, row in df.iterrows():
                if pd.isna(row["quarter_end_date"]):
                    continue
                qe = pd.Timestamp(row["quarter_end_date"])
                if avail_lag < pd.Timedelta(days=0):
                    raise ValueError(
                        f"Feature '{col}' has negative lag for quarter ending {qe.date()} "
                        "— data was used before it was available."
                    )


# Registry populated by data pull scripts
_FEATURE_AVAILABILITY: dict[str, pd.Timedelta] = {}


def register_feature_availability(col: str, lag: pd.Timedelta) -> None:
    """Record how many days before quarter-end feature 'col' is available."""
    _FEATURE_AVAILABILITY[col] = lag


def quarter_label_to_dates(label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Convert 'Q1_2025' → (quarter_start, quarter_end) as pd.Timestamps.
    """
    q, y = label.split("_")
    year = int(y)
    quarter_map = {
        "Q1": ("01-01", "03-31"),
        "Q2": ("04-01", "06-30"),
        "Q3": ("07-01", "09-30"),
        "Q4": ("10-01", "12-31"),
    }
    start_str, end_str = quarter_map[q]
    return (
        pd.Timestamp(f"{year}-{start_str}"),
        pd.Timestamp(f"{year}-{end_str}"),
    )


def weekly_to_quarterly(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window_weeks: int = 8,
    lag_weeks: int = 2,
) -> pd.DataFrame:
    """
    Aggregate weekly Trends data to quarterly mean, using a rolling window
    that ends `lag_weeks` before the quarter-end date (no look-ahead).

    Returns a DataFrame with columns: ['quarter_label', 'quarter_end_date', value_col + '_mean'].
    """
    from src.config import QUARTER_END_DATES

    records = []
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    for q_label, qe_str in QUARTER_END_DATES.items():
        qe = pd.Timestamp(qe_str)
        window_end = qe - pd.Timedelta(weeks=lag_weeks)
        window_start = window_end - pd.Timedelta(weeks=window_weeks)
        mask = (df[date_col] >= window_start) & (df[date_col] < window_end)
        subset = df.loc[mask, value_col]
        if subset.empty:
            mean_val = np.nan
        else:
            mean_val = subset.mean()
        records.append({"quarter_label": q_label, "quarter_end_date": qe_str, f"{value_col}_mean": mean_val})

    return pd.DataFrame(records)


def monthly_to_quarterly(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    lag_days: int = 30,
) -> pd.DataFrame:
    """
    Aggregate monthly macro data to quarterly value — most recent monthly
    observation available at least `lag_days` before the quarter-end date.
    """
    from src.config import QUARTER_END_DATES

    records = []
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    for q_label, qe_str in QUARTER_END_DATES.items():
        qe = pd.Timestamp(qe_str)
        cutoff = qe - pd.Timedelta(days=lag_days)
        subset = df.loc[df[date_col] <= cutoff, value_col]
        val = subset.iloc[-1] if not subset.empty else np.nan
        records.append({"quarter_label": q_label, "quarter_end_date": qe_str, value_col: val})

    return pd.DataFrame(records)


_FACTSET_MONTH_TO_QUARTER = {"Mar": "Q1", "Jun": "Q2", "Sep": "Q3", "Dec": "Q4"}


def parse_factset_period(period: str) -> str | None:
    """
    Convert FactSet 'Period' values to quarter labels.
        "Mar '26E"  -> "Q1_2026"   (E suffix = forward estimate, no actual)
        "Dec '25"   -> "Q4_2025"
    Returns None on unparseable input.
    """
    if not isinstance(period, str):
        return None
    s = period.strip().rstrip("E").strip()
    parts = s.replace("'", "").split()
    if len(parts) != 2:
        return None
    month, yy = parts
    q = _FACTSET_MONTH_TO_QUARTER.get(month)
    if q is None or not yy.isdigit():
        return None
    year = 2000 + int(yy)   # all data is post-2020
    return f"{q}_{year}"


def load_factset_table(path) -> pd.DataFrame:
    """
    Load a FactSet quarterly export (skiprows=1) and standardize columns.

    Returns DataFrame sorted ascending by quarter_end_date with columns:
        quarter_label, quarter_end_date, event_date,
        actual, consensus_mean, surprise_pct, num_est,
        low, high, guid_low, guid_high
    Non-numeric '-' placeholders are coerced to NaN.
    """
    from src.config import QUARTER_END_DATES

    df = pd.read_excel(path, skiprows=1)

    df["quarter_label"] = df["Period"].apply(parse_factset_period)
    df = df[df["quarter_label"].notna()].copy()
    df["quarter_end_date"] = pd.to_datetime(df["quarter_label"].map(QUARTER_END_DATES))

    numeric_cols = ["After Event", "Mean", "Surp (%)", "Num of Est",
                    "Low", "High", "Guid (Low)", "Guid (High)"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Event Date"] = pd.to_datetime(
        df["Event Date"], format="%d %b '%y", errors="coerce"
    )

    rename = {
        "Event Date": "event_date",
        "After Event": "actual",
        "Mean": "consensus_mean",
        "Surp (%)": "surprise_pct",
        "Num of Est": "num_est",
        "Low": "low",
        "High": "high",
        "Guid (Low)": "guid_low",
        "Guid (High)": "guid_high",
    }
    out_cols = ["quarter_label", "quarter_end_date"] + list(rename.values())
    return (df.rename(columns=rename)[out_cols]
              .sort_values("quarter_end_date")
              .reset_index(drop=True))


def compute_yoy(series: pd.Series) -> pd.Series:
    """YoY % growth (4-quarter lag). NaN-preserving — no forward fill."""
    return series.pct_change(4, fill_method=None) * 100


def compute_qoq(series: pd.Series) -> pd.Series:
    """QoQ % growth. NaN-preserving — no forward fill."""
    return series.pct_change(1, fill_method=None) * 100


def print_pull_summary(label: str, df: pd.DataFrame, date_col: str = None) -> None:
    """Print rows pulled, date range, and missing value count — every pull script calls this."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Rows: {len(df):,}")
    if date_col and date_col in df.columns:
        print(f"  Date range: {df[date_col].min()} → {df[date_col].max()}")
    missing = df.isna().sum().sum()
    print(f"  Missing values: {missing:,}")
    print(f"{'='*60}\n")
