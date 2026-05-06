"""
pull_weather.py — Weather demand boost index from Open-Meteo Historical API.

Role: Corroborating evidence only — not a model feature (see project spec §8i).
Causal chain: above-average precipitation / cold snaps → people stay home and
order delivery rather than going out → GOV demand tailwind.

API: https://archive-api.open-meteo.com/v1/archive (free, no auth required).
Pull: 2018-01-01 to today, daily precipitation + temperature, 10 DASH markets.

Baseline: 2018–2022 per market × calendar quarter.
Composite: weather_demand_boost_index = equal-weighted mean of per-market
           quarterly precipitation z-scores (positive = wetter = tailwind).

EDA checkpoint (§8i): if Pearson r vs. GOV YoY growth > 0.6, reconsider
using weather as a model feature. If r < 0.4, corroborating role confirmed.
"""

import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import (
    QUARTER_END_DATES, GOV_ACTUALS,
    DATA_RAW, OUTPUTS_FIGURES,
    CHART_DPI, COLORS, RANDOM_SEED,
    WEATHER_RAW_PATH, WEATHER_ANOMALY_PATH,
)
from src.utils import print_pull_summary

np.random.seed(RANDOM_SEED)

# ── Market definitions (Section 8i) ────────────────────────────────────────────
MARKETS: dict[str, dict[str, float]] = {
    "new_york_city": {"lat": 40.7128, "lon": -74.0060},
    "los_angeles":   {"lat": 34.0522, "lon": -118.2437},
    "chicago":       {"lat": 41.8781, "lon": -87.6298},
    "san_francisco": {"lat": 37.7749, "lon": -122.4194},
    "seattle":       {"lat": 47.6062, "lon": -122.3321},
    "boston":        {"lat": 42.3601, "lon": -71.0589},
    "washington_dc": {"lat": 38.9072, "lon": -77.0369},
    "houston":       {"lat": 29.7604, "lon": -95.3698},
    "miami":         {"lat": 25.7617, "lon": -80.1918},
    "atlanta":       {"lat": 33.7490, "lon": -84.3880},
}

# MSA populations (millions) — 2020 Census / 2022 ACS estimates.
# Used to compute population-weighted variant of the composite index.
# Source: US Census Bureau, 2022 ACS 1-year estimates, metro statistical areas.
MSA_POPULATIONS_M: dict[str, float] = {
    "new_york_city": 20.1,  # New York-Newark-Jersey City MSA
    "los_angeles":   13.2,  # Los Angeles-Long Beach-Anaheim MSA
    "chicago":        9.5,  # Chicago-Naperville-Elgin MSA
    "houston":        7.3,  # Houston-The Woodlands-Sugar Land MSA
    "washington_dc":  6.4,  # Washington-Arlington-Alexandria MSA
    "miami":          6.2,  # Miami-Fort Lauderdale-Pompano Beach MSA
    "atlanta":        6.1,  # Atlanta-Sandy Springs-Alpharetta MSA
    "boston":         4.9,  # Boston-Cambridge-Newton MSA
    "san_francisco":  4.7,  # San Francisco-Oakland-Berkeley MSA
    "seattle":        4.0,  # Seattle-Tacoma-Bellevue MSA
}

HISTORY_START = "2018-01-01"
BASELINE_START = pd.Timestamp("2018-01-01")
BASELINE_END = pd.Timestamp("2022-12-31")  # 5-year baseline

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
COLD_SNAP_C = -6.7   # 20°F
API_SLEEP = 2.0       # seconds between market calls (respect Open-Meteo rate limit)
RETRY_SLEEP = 15.0    # seconds to wait after any failure before retry

_MONTH_TO_CALQ = {
    1: "Q1", 2: "Q1", 3: "Q1",
    4: "Q2", 5: "Q2", 6: "Q2",
    7: "Q3", 8: "Q3", 9: "Q3",
    10: "Q4", 11: "Q4", 12: "Q4",
}


# ── Date helpers ────────────────────────────────────────────────────────────────

def _quarter_bounds(q_label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """'Q1_2025' → (2025-01-01, 2025-03-31)"""
    q, y = q_label.split("_")
    year = int(y)
    starts = {"Q1": f"{year}-01-01", "Q2": f"{year}-04-01",
               "Q3": f"{year}-07-01", "Q4": f"{year}-10-01"}
    ends   = {"Q1": f"{year}-03-31", "Q2": f"{year}-06-30",
               "Q3": f"{year}-09-30", "Q4": f"{year}-12-31"}
    return pd.Timestamp(starts[q]), pd.Timestamp(ends[q])


# ── API pull ────────────────────────────────────────────────────────────────────

def _coerce_floats(values: list, fill_na: float = np.nan) -> np.ndarray:
    """Convert a list (possibly containing None) to a float ndarray."""
    arr = np.array([float(v) if v is not None else np.nan for v in values])
    if not np.isnan(fill_na):
        arr = np.where(np.isnan(arr), fill_na, arr)
    return arr


def _pull_market(market: str, lat: float, lon: float) -> pd.DataFrame:
    """Pull daily weather for one market. Retry once after RETRY_SLEEP on failure."""
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": HISTORY_START,
        "end_date": today,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "America/New_York",
    }
    _empty = pd.DataFrame(columns=["date", "market", "precip_mm", "temp_max_c", "temp_min_c"])

    for attempt in range(2):
        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
            r.raise_for_status()
            daily = r.json()["daily"]
            df = pd.DataFrame({
                "date":       pd.to_datetime(daily["time"]),
                "market":     market,
                "precip_mm":  _coerce_floats(daily["precipitation_sum"], fill_na=0.0),
                "temp_max_c": _coerce_floats(daily["temperature_2m_max"]),
                "temp_min_c": _coerce_floats(daily["temperature_2m_min"]),
            })
            return df
        except Exception as e:
            if attempt == 0:
                print(f"\n    WARNING: {market} attempt 1 failed ({e}). "
                      f"Waiting {RETRY_SLEEP:.0f}s then retrying …")
                time.sleep(RETRY_SLEEP)
            else:
                print(f"    WARNING: {market} failed after 2 attempts — skipping.")
                return _empty
    return _empty


def pull_all_markets() -> tuple[pd.DataFrame, list[str]]:
    """
    Pull raw daily weather for all 10 markets.
    Returns (stacked_df, list_of_failed_markets).
    """
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for market, coords in MARKETS.items():
        print(f"  {market:<20s} …", end=" ", flush=True)
        df = _pull_market(market, coords["lat"], coords["lon"])
        if df.empty:
            failed.append(market)
            print("FAILED")
        else:
            dmin = df["date"].min().date()
            dmax = df["date"].max().date()
            print(f"{len(df):,} rows | {dmin} → {dmax}")
            frames.append(df)
        time.sleep(API_SLEEP)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), failed


# ── Baseline computation ────────────────────────────────────────────────────────

def _compute_baselines(raw_df: pd.DataFrame) -> dict:
    """
    Per market:
      p90: 90th percentile of ALL daily precip values within 2018-2022 baseline.
           Any rainy day above this threshold counts as "extreme".
      Per calendar quarter Q: mean and std of yearly-total precip over 2018-2022,
           and mean count of extreme-precip days per year.

    Returns nested dict keyed as baselines[market]['p90'] and
    baselines[market]['Q1']['precip_mean'], etc.
    """
    baselines: dict = {}

    for market in raw_df["market"].unique():
        mdf = raw_df[raw_df["market"] == market].copy()
        bl = mdf[(mdf["date"] >= BASELINE_START) & (mdf["date"] <= BASELINE_END)].copy()

        p90 = float(np.percentile(bl["precip_mm"].values, 90))

        bl["cal_quarter"] = bl["date"].dt.month.map(_MONTH_TO_CALQ)
        bl["year"] = bl["date"].dt.year
        bl["is_extreme"] = (bl["precip_mm"] > p90).astype(int)

        market_bl: dict = {"p90": p90}
        for q in ("Q1", "Q2", "Q3", "Q4"):
            qbl = bl[bl["cal_quarter"] == q]
            per_year_precip = qbl.groupby("year")["precip_mm"].sum()
            per_year_extreme = qbl.groupby("year")["is_extreme"].sum()

            n = len(per_year_precip)
            market_bl[q] = {
                "precip_mean":       float(per_year_precip.mean()) if n > 0 else np.nan,
                "precip_std":        float(per_year_precip.std(ddof=1)) if n > 1 else np.nan,
                "extreme_days_mean": float(per_year_extreme.mean()) if n > 0 else np.nan,
            }
        baselines[market] = market_bl

    return baselines


# ── Quarterly feature computation ───────────────────────────────────────────────

def _compute_quarterly_metrics(raw_df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """
    Per market × DASH fiscal quarter:
      precip_total_mm, extreme_precip_days, cold_snap_days, precip_anomaly_z
    """
    records: list[dict] = []

    for q_label in QUARTER_END_DATES:
        q_start, q_end = _quarter_bounds(q_label)
        cal_q = q_label.split("_")[0]

        for market in MARKETS:
            if market not in baselines:
                records.append({
                    "quarter_label": q_label, "market": market,
                    "precip_total_mm": np.nan, "extreme_precip_days": np.nan,
                    "cold_snap_days": np.nan, "precip_anomaly_z": np.nan,
                })
                continue

            mdf = raw_df[
                (raw_df["market"] == market) &
                (raw_df["date"] >= q_start) &
                (raw_df["date"] <= q_end)
            ]

            if mdf.empty:
                records.append({
                    "quarter_label": q_label, "market": market,
                    "precip_total_mm": np.nan, "extreme_precip_days": np.nan,
                    "cold_snap_days": np.nan, "precip_anomaly_z": np.nan,
                })
                continue

            bl = baselines[market]
            p90 = bl["p90"]
            bl_q = bl[cal_q]

            precip_total = float(mdf["precip_mm"].sum())
            extreme_days = int((mdf["precip_mm"] > p90).sum())
            cold_days = int((mdf["temp_min_c"] < COLD_SNAP_C).sum())

            bl_mean = bl_q["precip_mean"]
            bl_std = bl_q["precip_std"]
            if pd.isna(bl_std) or bl_std == 0:
                z = np.nan
            else:
                z = (precip_total - bl_mean) / bl_std

            records.append({
                "quarter_label":    q_label,
                "market":           market,
                "precip_total_mm":  precip_total,
                "extreme_precip_days": extreme_days,
                "cold_snap_days":   cold_days,
                "precip_anomaly_z": z,
            })

    return pd.DataFrame(records)


# ── Composite ───────────────────────────────────────────────────────────────────

def _build_composite(quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds two composite variants:
      weather_demand_boost_index      — equal-weighted mean of per-market z-scores
      weather_demand_boost_index_popwt — MSA-population-weighted mean of z-scores

    Weights for the pop-weighted variant are re-normalized to the subset of
    markets that have non-NaN z-scores for that quarter, so a failed pull
    doesn't silently reduce the index magnitude.
    """
    total_pop = sum(MSA_POPULATIONS_M.values())
    base_weights = {m: MSA_POPULATIONS_M[m] / total_pop for m in MARKETS}

    rows: list[dict] = []

    for q_label, qe_str in QUARTER_END_DATES.items():
        qdf = quarterly_df[quarterly_df["quarter_label"] == q_label]

        # Per-market z-scores (keyed by market name)
        market_z: dict[str, float] = {}
        for market in MARKETS:
            mrow = qdf[qdf["market"] == market]
            z = (
                float(mrow["precip_anomaly_z"].iloc[0])
                if not mrow.empty and not pd.isna(mrow["precip_anomaly_z"].iloc[0])
                else np.nan
            )
            market_z[market] = z

        available = {m: z for m, z in market_z.items() if not np.isnan(z)}

        # Equal-weighted composite
        wdbi = float(np.mean(list(available.values()))) if available else np.nan

        # Population-weighted composite (re-normalize weights to available markets)
        if available:
            avail_pop = sum(MSA_POPULATIONS_M[m] for m in available)
            wdbi_pw = float(
                sum(z * MSA_POPULATIONS_M[m] / avail_pop for m, z in available.items())
            )
        else:
            wdbi_pw = np.nan

        extreme_comp = int(qdf["extreme_precip_days"].fillna(0).sum())
        cold_comp = int(qdf["cold_snap_days"].fillna(0).sum())

        row: dict = {
            "quarter_label":                    q_label,
            "quarter_end_date":                 qe_str,
            "weather_demand_boost_index":       wdbi,
            "weather_demand_boost_index_popwt": wdbi_pw,
            "extreme_weather_days_composite":   extreme_comp,
            "cold_snap_days_composite":         cold_comp,
        }
        for market, z in market_z.items():
            row[f"{market}_precip_anomaly_z"] = z

        rows.append(row)

    market_cols = [f"{m}_precip_anomaly_z" for m in MARKETS]
    col_order = (
        ["quarter_label", "quarter_end_date",
         "weather_demand_boost_index", "weather_demand_boost_index_popwt"]
        + market_cols
        + ["extreme_weather_days_composite", "cold_snap_days_composite"]
    )
    df = pd.DataFrame(rows)
    return df[col_order]


# ── Q1 2026 analysis ────────────────────────────────────────────────────────────

def _q1_analysis(composite_df: pd.DataFrame) -> None:
    """Print Q1 2026 ranking vs. historical Q1 2021–2025 for both index variants."""
    q1_labels = [f"Q1_{y}" for y in range(2021, 2027)]
    q1_sub = composite_df[composite_df["quarter_label"].isin(q1_labels)].set_index("quarter_label")

    for col, label in [
        ("weather_demand_boost_index",       "Equal-weighted"),
        ("weather_demand_boost_index_popwt", "Pop-weighted  "),
    ]:
        vals = q1_sub[col].dropna().to_dict()
        if "Q1_2026" not in vals:
            print(f"\n  {label}: Q1 2026 data unavailable.")
            continue

        val_2026 = vals["Q1_2026"]
        sorted_q1 = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        rank = next(i + 1 for i, (lbl, _) in enumerate(sorted_q1) if lbl == "Q1_2026")
        n = len(sorted_q1)
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        interpretation = "tailwind" if val_2026 > 0.5 else ("headwind" if val_2026 < -0.5 else "neutral")

        print(f"\n  {label}  WDBI: {val_2026:+.3f}  |  "
              f"ranks {ordinal} wettest of last {n} Q1s  |  {interpretation}")

    # Detailed ranking table using equal-weight (primary)
    ew_vals = q1_sub["weather_demand_boost_index"].dropna().to_dict()
    pw_vals = q1_sub["weather_demand_boost_index_popwt"].dropna().to_dict()
    sorted_q1 = sorted(ew_vals.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Quarter':<10}  {'Equal-wt':>10}  {'Pop-wt':>10}  {'Δ':>8}")
    print("  " + "-" * 45)
    for lbl, ew in sorted_q1:
        pw = pw_vals.get(lbl, np.nan)
        delta = pw - ew if not np.isnan(pw) else np.nan
        marker = "  ← Q1 2026" if lbl == "Q1_2026" else ""
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "  NaN"
        print(f"  {lbl:<10}  {ew:>+10.3f}  {pw:>+10.3f}  {delta_str:>8}{marker}")


# ── GOV correlation preview ─────────────────────────────────────────────────────

def _gov_correlation_preview(composite_df: pd.DataFrame) -> dict[str, float]:
    """Compute Pearson r vs. GOV YoY growth for both index variants."""
    gov_yoy: dict[str, float] = {}
    for q_label, val in GOV_ACTUALS.items():
        if val is None:
            continue
        q, y = q_label.split("_")
        prior = GOV_ACTUALS.get(f"{q}_{int(y) - 1}")
        if prior and prior > 0:
            gov_yoy[q_label] = (val / prior - 1) * 100

    results: dict[str, float] = {}
    for col, label in [
        ("weather_demand_boost_index",       "Equal-weighted"),
        ("weather_demand_boost_index_popwt", "Pop-weighted  "),
    ]:
        merged = composite_df[["quarter_label", col]].copy()
        merged["gov_yoy"] = merged["quarter_label"].map(gov_yoy)
        valid = merged.dropna(subset=[col, "gov_yoy"])
        if len(valid) < 4:
            print(f"  {label}: not enough quarters for correlation (need ≥4).")
            continue
        r = float(valid[col].corr(valid["gov_yoy"]))
        results[col] = r
        print(f"  {label}  r = {r:+.3f}  (n={len(valid)})")

    return results


# ── Figure ──────────────────────────────────────────────────────────────────────

def _plot_q1_comparison(composite_df: pd.DataFrame) -> None:
    """Grouped bar chart: equal-weighted vs pop-weighted WDBI for Q1 2021–2026."""
    q1_labels = [f"Q1_{y}" for y in range(2021, 2027)]
    q1_df = (
        composite_df[composite_df["quarter_label"].isin(q1_labels)]
        .set_index("quarter_label")
        .reindex(q1_labels)
    )

    ew_vals = q1_df["weather_demand_boost_index"].values
    pw_vals = q1_df["weather_demand_boost_index_popwt"].values
    years = [lbl.split("_")[1] for lbl in q1_labels]
    x = np.arange(len(years))
    w = 0.38

    # Q1 2026 bars get the forecast color; historical get primary/muted variant
    def _bar_color(lbl: str, is_popwt: bool) -> str:
        if lbl == "Q1_2026":
            return COLORS["forecast"] if not is_popwt else "#C45E20"  # darker amber for pop-wt
        return COLORS["dash_primary"] if not is_popwt else "#A0231A"  # darker red for pop-wt

    ew_colors = [_bar_color(lbl, False) for lbl in q1_labels]
    pw_colors = [_bar_color(lbl, True)  for lbl in q1_labels]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")

    bars_ew = ax.bar(x - w / 2, ew_vals, width=w, color=ew_colors, zorder=3, label="_nolegend_")
    bars_pw = ax.bar(x + w / 2, pw_vals, width=w, color=pw_colors, zorder=3, label="_nolegend_")

    ax.axhline(0, color="#444444", linewidth=1.2, zorder=2)
    ax.grid(axis="y", alpha=0.3, zorder=1)

    for bar, v in list(zip(bars_ew, ew_vals)) + list(zip(bars_pw, pw_vals)):
        if np.isnan(v):
            continue
        y_off = 0.04 if v >= 0 else -0.09
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + y_off,
                f"{v:+.2f}", ha="center", va=va, fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Weather Demand Boost Index (z-score)", fontsize=11)
    ax.set_title(
        "Q1 Weather Demand Boost Index vs. 5-Year Baseline\n"
        "Top-10 DASH Markets  |  Equal-Weighted vs. MSA Population-Weighted  |  2021–2026",
        fontsize=12, fontweight="bold",
    )

    legend_elements = [
        Patch(facecolor=COLORS["dash_primary"], label="Equal-weighted (historical)"),
        Patch(facecolor="#A0231A",              label="Pop-weighted (historical)"),
        Patch(facecolor=COLORS["forecast"],     label="Equal-weighted Q1 2026"),
        Patch(facecolor="#C45E20",              label="Pop-weighted Q1 2026"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.85, fontsize=8.5,
              ncol=2)

    ax.text(0.02, 0.03,
            "Positive = above-average precipitation = demand tailwind  |  "
            "Negative = demand headwind",
            transform=ax.transAxes, fontsize=8, color="#666666", va="bottom")

    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig_path = OUTPUTS_FIGURES / "weather_q1_2026.png"
    fig.savefig(fig_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")


# ── Main ────────────────────────────────────────────────────────────────────────

def pull_weather() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline:
      1. Pull raw daily weather for all 10 markets
      2. Save raw CSV
      3. Compute 5-year baselines (2018-2022)
      4. Compute quarterly metrics per market
      5. Build composite index
      6. Save weather_anomaly.csv
      7. Print summary table + Q1 2026 analysis + GOV correlation preview
      8. Generate Q1 comparison figure
    Returns (raw_df, composite_df).
    """
    print("\n" + "=" * 60)
    print("  Weather Pull — Open-Meteo Historical API (free, no auth)")
    print("  Role: corroborating evidence only (not a model feature)")
    print("=" * 60)

    # 1. Pull
    print("\n--- Market Data Pull ---")
    raw_df, failed = pull_all_markets()

    if raw_df.empty:
        print("\nERROR: All market pulls failed. signal_available = False.")
        return pd.DataFrame(), pd.DataFrame()

    if failed:
        print(f"\n  Markets where pull failed: {failed}")

    # 2. Save raw
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(WEATHER_RAW_PATH, index=False)
    print(f"\n  Raw data saved → {WEATHER_RAW_PATH}")
    print_pull_summary("Weather Raw (daily, all markets)", raw_df, "date")

    # 3. Baselines
    print("--- 5-Year Baselines (2018–2022) ---")
    baselines = _compute_baselines(raw_df)
    for market, bl in baselines.items():
        p90 = bl["p90"]
        print(
            f"  {market:<20s}  p90={p90:5.1f}mm | "
            f"Q1_mean={bl['Q1']['precip_mean']:5.0f}mm  "
            f"Q4_mean={bl['Q4']['precip_mean']:5.0f}mm"
        )

    # 4. Quarterly metrics
    print("\n--- Quarterly Metrics per Market ---")
    quarterly_df = _compute_quarterly_metrics(raw_df, baselines)

    # 5. Composite
    composite_df = _build_composite(quarterly_df)

    # 6. Save anomaly CSV
    composite_df.to_csv(WEATHER_ANOMALY_PATH, index=False)
    print(f"\n  Anomaly data saved → {WEATHER_ANOMALY_PATH}")
    print_pull_summary("Weather Anomaly (quarterly)", composite_df, "quarter_end_date")

    # 7a. Summary table
    print("--- Weather Demand Boost Index — DASH Quarters ---")
    print(f"  {'Quarter':<12}  {'Equal-wt':>10}  {'Pop-wt':>10}  {'Δ':>7}  "
          f"{'Extreme Days':>14}  {'Cold Days':>11}")
    print("  " + "-" * 73)
    for _, row in composite_df.iterrows():
        ew  = row["weather_demand_boost_index"]
        pw  = row["weather_demand_boost_index_popwt"]
        ew_s = f"{ew:+.3f}" if not pd.isna(ew) else "   NaN"
        pw_s = f"{pw:+.3f}" if not pd.isna(pw) else "   NaN"
        d_s  = f"{pw - ew:+.3f}" if (not pd.isna(ew) and not pd.isna(pw)) else "   NaN"
        print(
            f"  {row['quarter_label']:<12}  {ew_s:>10}  {pw_s:>10}  {d_s:>7}  "
            f"{int(row['extreme_weather_days_composite']):>14}  "
            f"{int(row['cold_snap_days_composite']):>11}"
        )

    # 7b. Q1 2026 analysis
    print("\n--- Q1 2026 Specific Analysis ---")
    _q1_analysis(composite_df)

    # 7c. GOV correlation preview
    print("\n--- GOV Correlation Preview (preliminary — full EDA in notebook) ---")
    r_dict = _gov_correlation_preview(composite_df)
    for col, r in r_dict.items():
        if abs(r) > 0.6:
            print(f"  NOTE: {col} r = {r:.3f} > 0.6 — "
                  "reconsider including as model feature (EDA §8i).")
        else:
            print(f"  NOTE: {col} r = {r:.3f} — "
                  "corroborating evidence role confirmed.")
    print(
        "\n  Reminder: If EDA correlation r > 0.6, reconsider including weather "
        "as a model feature. If r < 0.4, corroborating evidence role confirmed."
    )

    # 8. Figure
    print("\n--- Q1 Comparison Figure ---")
    _plot_q1_comparison(composite_df)

    signal_available = len(failed) < len(MARKETS)
    print(f"\n  signal_available = {signal_available}  "
          f"({len(MARKETS) - len(failed)}/{len(MARKETS)} markets pulled successfully)")

    return raw_df, composite_df


def save_weather() -> None:
    raw_df, composite_df = pull_weather()
    if not composite_df.empty:
        print(f"\n  Raw shape:       {raw_df.shape}")
        print(f"  Anomaly shape:   {composite_df.shape}")
        print(f"  Anomaly columns ({len(composite_df.columns)}): {list(composite_df.columns)}")


if __name__ == "__main__":
    save_weather()
