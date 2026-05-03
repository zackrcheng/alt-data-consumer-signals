"""
event_study.py — CRSP-based earnings event study for DASH.

Computes:
  CAR[-1, +2]  (primary window)
  CAR[0, +1]   (tight window)

Abnormal return = DASH daily return − value-weighted market return (crsp.dsi).

Fallback: if CRSP unavailable, use yfinance DASH returns minus SPY — flag as approximation.

Output: data/raw/crsp_event_study.csv with CAR columns and GOV surprise for each event.
        outputs/figures/event_study_scatter.png (Exhibit 6)

See CLAUDE.md §13 for full spec.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
matplotlib.rcParams.update({"figure.facecolor": "white"})

from src.config import (
    EARNINGS_DATES, CAR_WINDOWS, DASH_GOV_MASTER_PATH,
    CRSP_EVENT_STUDY_PATH, PRICES_DAILY_PATH,
    OUTPUTS_FIGURES, OUTPUTS_TABLES, CHART_DPI, COLORS,
)


def compute_cars_crsp() -> pd.DataFrame:
    """
    Compute CARs using CRSP daily returns and value-weighted market returns.
    Requires WRDS connection (WRDS_USERNAME in .env).
    """
    try:
        import wrds
    except ImportError:
        raise ImportError("wrds not installed.")

    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise EnvironmentError("WRDS_USERNAME not in .env")

    db = wrds.Connection(wrds_username=username)

    # Look up DASH permno
    permno_query = """
        SELECT permno FROM crsp.dsenames
        WHERE ticker = 'DASH'
        ORDER BY nameendt DESC LIMIT 1
    """
    try:
        permno_df = db.raw_sql(permno_query)
        dash_permno = int(permno_df.iloc[0]["permno"])
    except Exception as e:
        print(f"  CRSP permno lookup failed: {e}. Falling back to yfinance.")
        db.close()
        return compute_cars_yfinance()

    # Pull daily returns for DASH and value-weighted market around each event
    records = []
    gov = pd.read_csv(DASH_GOV_MASTER_PATH)

    for quarter, event_date_str in EARNINGS_DATES.items():
        event_date = pd.Timestamp(event_date_str)
        window_start = event_date - pd.Timedelta(days=10)
        window_end = event_date + pd.Timedelta(days=10)

        try:
            dash_ret_q = f"""
                SELECT date, ret
                FROM crsp.dsf
                WHERE permno = {dash_permno}
                  AND date BETWEEN '{window_start.date()}' AND '{window_end.date()}'
                ORDER BY date
            """
            mkt_ret_q = f"""
                SELECT date, vwretd
                FROM crsp.dsi
                WHERE date BETWEEN '{window_start.date()}' AND '{window_end.date()}'
                ORDER BY date
            """
            dash_ret = db.raw_sql(dash_ret_q)
            mkt_ret = db.raw_sql(mkt_ret_q)
        except Exception as e:
            print(f"  CRSP query failed for {quarter}: {e}")
            continue

        dash_ret["date"] = pd.to_datetime(dash_ret["date"])
        mkt_ret["date"] = pd.to_datetime(mkt_ret["date"])
        merged = dash_ret.merge(mkt_ret, on="date", how="inner")
        merged["abnormal_ret"] = merged["ret"] - merged["vwretd"]
        merged = merged.sort_values("date")

        # Identify event date index in trading days
        trading_days = merged["date"].tolist()
        if event_date not in trading_days:
            # Find nearest trading day
            diffs = [abs((d - event_date).days) for d in trading_days]
            event_idx = diffs.index(min(diffs))
        else:
            event_idx = trading_days.index(event_date)

        def car(window):
            lo, hi = window
            start_i = max(0, event_idx + lo)
            end_i = min(len(merged) - 1, event_idx + hi)
            return merged.iloc[start_i:end_i + 1]["abnormal_ret"].sum()

        # GOV surprise for this quarter
        gov_row = gov[gov["quarter_label"] == quarter]
        gov_surprise = gov_row["gov_surprise_pct"].iloc[0] if not gov_row.empty else np.nan

        records.append({
            "quarter_label": quarter,
            "earnings_date": event_date_str,
            "car_minus1_plus2": car(CAR_WINDOWS["primary"]),
            "car_0_plus1": car(CAR_WINDOWS["tight"]),
            "gov_surprise_pct": gov_surprise,
            "source": "CRSP",
        })

    db.close()
    return pd.DataFrame(records)


def compute_cars_yfinance() -> pd.DataFrame:
    """
    Fallback: compute CARs using yfinance DASH and SPY daily returns.
    Less clean than CRSP — flag as approximation in write-up.
    """
    import yfinance as yf

    print("  Using yfinance fallback for event study (CRSP unavailable).")
    gov = pd.read_csv(DASH_GOV_MASTER_PATH)

    # Download DASH and SPY returns
    all_tickers = ["DASH", "SPY"]
    raw = yf.download(all_tickers, start="2023-01-01", auto_adjust=True, progress=False)
    prices = raw["Close"]
    returns = prices.pct_change()

    records = []
    for quarter, event_date_str in EARNINGS_DATES.items():
        event_date = pd.Timestamp(event_date_str)
        trading_days = returns.index.tolist()

        if event_date not in trading_days:
            diffs = [abs((d - event_date).days) for d in trading_days]
            event_idx = diffs.index(min(diffs))
        else:
            event_idx = trading_days.index(event_date)

        def car_yfin(window):
            lo, hi = window
            start_i = max(0, event_idx + lo)
            end_i = min(len(returns) - 1, event_idx + hi)
            abnormal = returns["DASH"].iloc[start_i:end_i + 1] - returns["SPY"].iloc[start_i:end_i + 1]
            return abnormal.sum()

        gov_row = gov[gov["quarter_label"] == quarter]
        gov_surprise = gov_row["gov_surprise_pct"].iloc[0] if not gov_row.empty else np.nan

        records.append({
            "quarter_label": quarter,
            "earnings_date": event_date_str,
            "car_minus1_plus2": car_yfin(CAR_WINDOWS["primary"]),
            "car_0_plus1": car_yfin(CAR_WINDOWS["tight"]),
            "gov_surprise_pct": gov_surprise,
            "source": "yfinance_approx",
        })

    return pd.DataFrame(records)


def plot_event_study_scatter(df: pd.DataFrame, q1_2026_surprise_forecast: float = None) -> None:
    """
    Exhibit 6: GOV surprise % (x) vs. CAR[-1,+2] (y).
    Marks Q1 2026 predicted surprise as forward-looking point.
    """
    import statsmodels.api as sm

    avail = df.dropna(subset=["gov_surprise_pct", "car_minus1_plus2"])
    if len(avail) < 3:
        print("  Insufficient data for event study scatter.")
        return

    X = sm.add_constant(avail[["gov_surprise_pct"]])
    y = avail["car_minus1_plus2"] * 100   # convert to %
    model = sm.OLS(y, X).fit()
    r2 = model.rsquared
    beta = model.params.get("gov_surprise_pct", np.nan)
    pval = model.pvalues.get("gov_surprise_pct", np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(avail["gov_surprise_pct"], y, color=COLORS["dash_primary"],
               s=80, zorder=3, label="Historical earnings events")

    # Regression line
    x_range = np.linspace(avail["gov_surprise_pct"].min() - 1, avail["gov_surprise_pct"].max() + 1, 100)
    y_hat = model.params["const"] + beta * x_range
    ax.plot(x_range, y_hat, color=COLORS["dash_primary"], linewidth=1.5, linestyle="--",
            label=f"OLS fit (β={beta:.2f}, R²={r2:.2f}, p={pval:.2f})")

    # Annotate quarters
    for _, row in avail.iterrows():
        ax.annotate(row["quarter_label"], (row["gov_surprise_pct"], row["car_minus1_plus2"] * 100),
                    fontsize=7, ha="left", va="bottom")

    # Q1 2026 forward-looking marker
    if q1_2026_surprise_forecast is not None:
        implied_car = model.params["const"] + beta * q1_2026_surprise_forecast
        ax.axvline(x=q1_2026_surprise_forecast, color=COLORS["forecast"], linestyle=":",
                   linewidth=1.5, label=f"Q1 2026 model forecast ({q1_2026_surprise_forecast:+.1f}pp)")
        ax.axhspan(implied_car - 2, implied_car + 2, alpha=0.12, color=COLORS["forecast"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("GOV Surprise vs. Consensus (%)", fontsize=11)
    ax.set_ylabel("CAR [-1, +2] (%)", fontsize=11)
    ax.set_title("Exhibit 6 — GOV Surprise vs. Abnormal Return (DASH Earnings)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out_path = OUTPUTS_FIGURES / "exhibit6_event_study.png"
    fig.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def run_event_study(q1_2026_surprise: float = None) -> pd.DataFrame:
    """Full event study pipeline: compute CARs, save, plot."""
    try:
        df = compute_cars_crsp()
    except Exception as e:
        print(f"  CRSP failed ({e}). Using yfinance fallback.")
        df = compute_cars_yfinance()

    df.to_csv(CRSP_EVENT_STUDY_PATH, index=False)
    print(f"Saved: {CRSP_EVENT_STUDY_PATH}")

    plot_event_study_scatter(df, q1_2026_surprise_forecast=q1_2026_surprise)
    return df


if __name__ == "__main__":
    run_event_study()
