"""
pull_appstore.py — App store review velocity + VADER sentiment analysis.

NOT foot traffic — foot traffic measures physical store visits and has no
credible causal chain to delivery GOV (see project spec §8f). Review velocity
is an engagement-side signal: downloads and post-order feedback.

Data sources:
  Google Play (primary, historical)
    - Reviews pulled sorted NEWEST; text included for VADER sentiment
    - Limits sized to reach DASH IPO (Dec 2020) with 25% buffer:
      DASH 270K, UBER 290K, INSTACART 90K, GRUBHUB 60K, GOPUFF 50K
    - Total runtime: ~30 min
  iTunes lookup (current week only)
    - Total iOS ratings count per app — snapshot, no historical API
  iTunes RSS (current week only)
    - Food & Drink category rank — snapshot only
    - ~500 recent reviews for current-week iOS sentiment

Per-review fields collected from Google Play:
  date            created timestamp
  score           1–5 star rating
  text            full review body (for VADER)
  thumbsUpCount   upvoted reviews carry more weight

Per-review computed:
  vader_compound  VADER compound score on text (-1 to +1)
  is_complaint    1 if score<=2 OR vader_compound<-0.2
  is_positive     1 if score>=4 AND vader_compound>0.2

Weekly aggregates per app (prefix: dash / uber / cart / grubhub / gopuff):
  Volume:
    {p}_review_count             total reviews posted that week
    {p}_review_velocity_wow_pct  WoW % change in review count
  Sentiment:
    {p}_mean_star                simple average star rating
    {p}_mean_vader               VADER compound score mean
    {p}_complaint_ratio          is_complaint / total reviews
    {p}_positive_ratio           is_positive / total reviews
    {p}_net_sentiment            positive_ratio - complaint_ratio
    {p}_weighted_sentiment       thumbsUpCount-weighted mean vader
  Combined:
    {p}_engagement_x_sentiment   review_count * net_sentiment
                                 positive = high engagement + good sentiment
                                 negative = high engagement + bad sentiment

Cross-app derived features:
  --- Velocity ratios ---
  dash_vs_uber_review_ratio       DASH / UBER review count
  dash_vs_cart_review_ratio       DASH / INSTACART review count
  dash_vs_grubhub_review_ratio    DASH / GRUBHUB review count

  --- Volume share ---
  three_way_appstore_share        DASH / (DASH + UBER + CART)
  four_way_appstore_share         DASH / (DASH + UBER + CART + GRUBHUB)
  five_way_appstore_share         DASH / (DASH + UBER + CART + GRUBHUB + GOPUFF)

  --- Sentiment comparisons ---
  dash_vs_uber_net_sentiment      DASH - UBER net_sentiment delta
  dash_vs_uber_complaint_delta    DASH - UBER complaint_ratio (negative = fewer DASH complaints)

  --- Positive-volume share ---
  three_way_positive_share        DASH positive reviews / (DASH+UBER+CART) positive reviews

Aggregation (no look-ahead): 8-week window ending 2 weeks before quarter-end.
Quarterly features assembled in build_master_df.py.
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from google_play_scraper import Sort
from google_play_scraper import reviews as gp_reviews_fn

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import APPS, APPSTORE_PATH, RANDOM_SEED
from src.utils import print_pull_summary

np.random.seed(RANDOM_SEED)

_vader = SentimentIntensityAnalyzer()

# Per-app GP review limits sized to reach back to the DASH IPO (Dec 10 2020),
# which is the hard floor for the GOV target variable (project spec §5).
#
# Limits were calibrated empirically after a first pull revealed that
# historical review rates (2021 COVID growth era) were higher than today:
#   DASH:      ~148/day avg → 270K only reached May-2021; 340K targets Dec-2020
#   UBER_EATS: ~160/day avg → 290K only reached Dec-2021; 420K targets Dec-2020
#   INSTACART:  ~48/day avg →  90K only reached Mar-2021; 100K targets Dec-2020
#   GRUBHUB:    ~33/day avg →  60K reached Dec-2020 (11d gap); 70K adds buffer
#   GOPUFF:      ~7/day avg →  50K covers back to 2013 — no change needed
#
# Total runtime on re-pull: ~1.5 hour.
MAX_GP_REVIEWS_BY_APP: dict[str, int] = {
    "DASH":       380_000,   # 270K reached May-2021; +110K closes gap to Dec-2020 IPO
    "UBER_EATS":  440_000,   # 290K reached Dec-2021; +150K closes gap to Dec-2020
    "INSTACART":  120_000,   # 90K reached Mar-2021;  +30K closes gap to Dec-2020
    "GRUBHUB":     70_000,   # 60K reached Dec-2020 (11d gap); +10K buffer
    "GOPUFF":      50_000,   # already covers back to 2013 — no change needed
}
GP_BATCH_SIZE = 200
GP_SLEEP_SEC = 0.5

ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup?id={app_id}&country=us"
ITUNES_RANK_URL = (
    "https://itunes.apple.com/us/rss/topfreeapplications/limit=200/genre=6023/json"
)
ITUNES_REVIEWS_URL = (
    "https://itunes.apple.com/us/rss/customerreviews/page={page}"
    "/id={app_id}/sortBy=mostRecent/json"
)

# Short prefix for column names — INSTACART = 'cart' to match thesis language
APP_PREFIX = {
    "DASH":      "dash",
    "UBER_EATS": "uber",
    "INSTACART": "cart",
    "GRUBHUB":   "grubhub",
    "GOPUFF":    "gopuff",
}


# ── Google Play: pull reviews ──────────────────────────────────────────────────

def _pull_gp_reviews_for_app(app_key: str, max_reviews: int | None = None) -> pd.DataFrame:
    """
    Pull GP reviews (sorted NEWEST) for one app.
    Collects date, score, text, thumbsUpCount — text needed for VADER.
    Returns DataFrame; empty DataFrame on failure.
    """
    if max_reviews is None:
        max_reviews = MAX_GP_REVIEWS_BY_APP[app_key]
    google_id = APPS[app_key]["google"]
    all_reviews: list[dict] = []
    token = None
    batches = (max_reviews + GP_BATCH_SIZE - 1) // GP_BATCH_SIZE

    for i in range(batches):
        try:
            result, token = gp_reviews_fn(
                google_id,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=GP_BATCH_SIZE,
                continuation_token=token,
            )
        except Exception as e:
            print(f"    GP batch {i+1} failed for {app_key}: {e}. Stopping.")
            break

        if not result:
            break

        # Keep only the four fields needed — avoids storing megabytes of metadata
        all_reviews.extend(
            {
                "date": r["at"],
                "score": r.get("score", np.nan),
                "text": r.get("content") or "",
                "thumbsUpCount": r.get("thumbsUpCount") or 0,
            }
            for r in result
        )

        if (i + 1) % 50 == 0:
            print(f"    {app_key}: {len(all_reviews):,} reviews …", flush=True)

        if token is None:
            break
        time.sleep(GP_SLEEP_SEC)

    if not all_reviews:
        print(f"  WARNING: no GP reviews returned for {app_key}.")
        return pd.DataFrame(columns=["date", "score", "text", "thumbsUpCount"])

    df = pd.DataFrame(all_reviews)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── VADER sentiment ────────────────────────────────────────────────────────────

def _add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute vader_compound, is_complaint, is_positive for each review.
    VADER is rule-based and processes ~300K texts/second — no training needed.
    """
    texts = df["text"].fillna("").tolist()
    compounds = [_vader.polarity_scores(t)["compound"] for t in texts]

    df = df.copy()
    df["vader_compound"] = compounds
    # is_complaint: bad star OR clearly negative text (union = more recall)
    df["is_complaint"] = (
        (df["score"].fillna(3) <= 2) | (df["vader_compound"] < -0.2)
    ).astype(int)
    # is_positive: good star AND clearly positive text (intersection = more precision)
    df["is_positive"] = (
        (df["score"].fillna(3) >= 4) & (df["vader_compound"] > 0.2)
    ).astype(int)
    return df


# ── Weekly aggregation ─────────────────────────────────────────────────────────

def _agg_to_weekly(df: pd.DataFrame, app_key: str) -> pd.DataFrame:
    """
    Aggregate per-review DataFrame to weekly signals.
    Returns wide DataFrame with 'week' column and all per-app signals prefixed.
    """
    df = df.copy()
    # W period: ISO week (Mon–Sun); start_time is Monday — matches W-MON date_range
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    grp = df.groupby("week")

    # --- Volume ---
    review_count = grp.size().rename("review_count")

    # --- Sentiment ---
    mean_star     = grp["score"].mean().rename("mean_star")
    mean_vader    = grp["vader_compound"].mean().rename("mean_vader")
    complaint_r   = grp["is_complaint"].mean().rename("complaint_ratio")
    positive_r    = grp["is_positive"].mean().rename("positive_ratio")

    # thumbsUpCount-weighted mean VADER (upvoted reviews are more representative)
    def _weighted_vader(g: pd.DataFrame) -> float:
        w = (g["thumbsUpCount"].fillna(0).clip(lower=0) + 1).values
        return float(np.average(g["vader_compound"].values, weights=w))

    weighted_s = grp.apply(_weighted_vader, include_groups=False).rename("weighted_sentiment")

    # --- Positive review count for positive_share cross-app metric ---
    positive_count = grp["is_positive"].sum().rename("positive_count")

    weekly = pd.concat(
        [review_count, mean_star, mean_vader, complaint_r, positive_r,
         weighted_s, positive_count],
        axis=1,
    ).reset_index()
    weekly.rename(columns={"week": "date"}, inplace=True)

    # Derived within-app
    weekly["net_sentiment"] = weekly["positive_ratio"] - weekly["complaint_ratio"]
    weekly["engagement_x_sentiment"] = weekly["review_count"] * weekly["net_sentiment"]

    # WoW % change in review count (before gap-filling so zeros don't distort)
    weekly = weekly.sort_values("date")
    weekly["review_velocity_wow_pct"] = weekly["review_count"].pct_change() * 100

    # Fill sparse weeks within observed range with NaN (not 0 — cleaner for model)
    full_range = pd.date_range(weekly["date"].min(), weekly["date"].max(), freq="W-MON")
    weekly = (
        weekly.set_index("date")
        .reindex(full_range)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Prefix all metric columns with the short app name
    prefix = APP_PREFIX[app_key]
    weekly.rename(
        columns={c: f"{prefix}_{c}" for c in weekly.columns if c != "date"},
        inplace=True,
    )
    return weekly


# ── iTunes: current-week snapshot ─────────────────────────────────────────────

def _pull_itunes_ios_reviews(app_id: int, max_pages: int = 10) -> pd.DataFrame:
    """
    Pull up to 500 recent iOS reviews via iTunes Customer Reviews RSS (10 pages × 50).
    Only covers ~1 week of data — used for current-week iOS sentiment snapshot.
    """
    rows = []
    for page in range(1, max_pages + 1):
        try:
            url = ITUNES_REVIEWS_URL.format(page=page, app_id=app_id)
            r = requests.get(url, timeout=10)
            data = r.json()
            entries = data.get("feed", {}).get("entry", [])
            if not entries:
                break
            for e in entries:
                if not isinstance(e, dict) or "im:rating" not in e:
                    continue  # first entry is app metadata, not a review
                rows.append({
                    "date": pd.to_datetime(e["updated"]["label"], utc=True).tz_localize(None),
                    "score": int(e["im:rating"]["label"]),
                    "text": (e.get("content", {}) or {}).get("label", ""),
                    "thumbsUpCount": 0,  # iTunes RSS doesn't expose upvote counts
                })
        except Exception as e:
            print(f"    iTunes RSS page {page} failed: {e}")
            break
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame(columns=["date", "score", "text", "thumbsUpCount"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def pull_itunes_snapshots() -> tuple[dict, dict]:
    """
    Pull current iOS ratings count (lookup API) and Food & Drink rank (RSS).
    Returns ({app_key: ratings_count}, {app_key: rank}).
    iTunes historical data unavailable via free APIs — snapshot only.
    """
    # Category ranks
    ranks: dict[str, int | None] = {k: None for k in APPS}
    try:
        r = requests.get(ITUNES_RANK_URL, timeout=15)
        entries = r.json().get("feed", {}).get("entry", [])
        id_to_rank = {
            e.get("id", {}).get("attributes", {}).get("im:id"): idx + 1
            for idx, e in enumerate(entries)
        }
        for app_key, meta in APPS.items():
            ranks[app_key] = id_to_rank.get(str(meta["ios"]))
    except Exception as e:
        print(f"  iTunes RSS rank fetch failed: {e}")

    # Total ratings counts
    ratings_counts: dict[str, int | None] = {}
    for app_key, meta in APPS.items():
        try:
            r = requests.get(ITUNES_LOOKUP_URL.format(app_id=meta["ios"]), timeout=10)
            results = r.json().get("results", [])
            ratings_counts[app_key] = results[0].get("userRatingCount") if results else None
        except Exception as e:
            print(f"  iTunes lookup failed for {app_key}: {e}")
            ratings_counts[app_key] = None
        time.sleep(0.3)

    return ratings_counts, ranks


# ── Cross-app derived features ─────────────────────────────────────────────────

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_cross_app_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add velocity ratios, volume share metrics, sentiment comparisons, and
    positive-volume share from per-app weekly columns.
    """
    d = df.copy()

    # Short handles for review count and derived columns
    dash_ct  = "dash_review_count"
    uber_ct  = "uber_review_count"
    cart_ct  = "cart_review_count"
    grub_ct  = "grubhub_review_count"
    gopu_ct  = "gopuff_review_count"

    # --- Velocity ratios ---
    if {dash_ct, uber_ct} <= set(d.columns):
        d["dash_vs_uber_review_ratio"] = _safe_div(d[dash_ct], d[uber_ct])
    if {dash_ct, cart_ct} <= set(d.columns):
        d["dash_vs_cart_review_ratio"] = _safe_div(d[dash_ct], d[cart_ct])
    if {dash_ct, grub_ct} <= set(d.columns):
        d["dash_vs_grubhub_review_ratio"] = _safe_div(d[dash_ct], d[grub_ct])

    # --- Volume share ---
    three = [c for c in [dash_ct, uber_ct, cart_ct] if c in d.columns]
    if len(three) == 3:
        d["three_way_appstore_share"] = _safe_div(d[dash_ct], d[three].sum(axis=1))

    four = [c for c in [dash_ct, uber_ct, cart_ct, grub_ct] if c in d.columns]
    if len(four) == 4:
        d["four_way_appstore_share"] = _safe_div(d[dash_ct], d[four].sum(axis=1))

    five = [c for c in [dash_ct, uber_ct, cart_ct, grub_ct, gopu_ct] if c in d.columns]
    if len(five) == 5:
        d["five_way_appstore_share"] = _safe_div(d[dash_ct], d[five].sum(axis=1))

    # --- Sentiment comparisons ---
    if {"dash_net_sentiment", "uber_net_sentiment"} <= set(d.columns):
        d["dash_vs_uber_net_sentiment"] = d["dash_net_sentiment"] - d["uber_net_sentiment"]
    if {"dash_complaint_ratio", "uber_complaint_ratio"} <= set(d.columns):
        # Negative value = DASH has fewer complaints than UBER (good for DASH)
        d["dash_vs_uber_complaint_delta"] = (
            d["dash_complaint_ratio"] - d["uber_complaint_ratio"]
        )

    # --- Positive-volume share (of all positive reviews, what % are for DASH?) ---
    dash_pos = "dash_positive_count"
    uber_pos = "uber_positive_count"
    cart_pos = "cart_positive_count"
    three_pos = [c for c in [dash_pos, uber_pos, cart_pos] if c in d.columns]
    if len(three_pos) == 3:
        total_pos = d[three_pos].sum(axis=1).replace(0, np.nan)
        d["three_way_positive_share"] = d[dash_pos] / total_pos

    return d


# ── Attach iTunes snapshot to current week ─────────────────────────────────────

def _attach_itunes_snapshot(
    weekly_df: pd.DataFrame,
    ratings_counts: dict,
    ranks: dict,
    ios_reviews_by_app: dict[str, pd.DataFrame],
    snapshot_date: str,
) -> pd.DataFrame:
    """
    Insert iOS ratings count, rank, and current-week sentiment into weekly_df.
    All iOS columns are NaN for historical weeks — iTunes has no free historical API.
    """
    today_week = pd.to_datetime(snapshot_date).to_period("W").start_time

    # Build iOS sentiment for current week from iTunes RSS reviews
    ios_weekly_rows = []
    for app_key, ios_df in ios_reviews_by_app.items():
        if ios_df.empty:
            continue
        prefix = APP_PREFIX[app_key]
        ios_df = _add_vader_sentiment(ios_df)
        ios_weekly_rows.append({
            "date": today_week,
            f"{prefix}_ios_review_count":   len(ios_df),
            f"{prefix}_ios_mean_star":       ios_df["score"].mean(),
            f"{prefix}_ios_mean_vader":      ios_df["vader_compound"].mean(),
            f"{prefix}_ios_complaint_ratio": ios_df["is_complaint"].mean(),
            f"{prefix}_ios_positive_ratio":  ios_df["is_positive"].mean(),
            f"{prefix}_ios_net_sentiment":   ios_df["is_positive"].mean() - ios_df["is_complaint"].mean(),
        })

    if ios_weekly_rows:
        ios_snapshot = pd.DataFrame(ios_weekly_rows)
        # Merge all apps into one row for today's week
        ios_row = {"date": today_week}
        for row in ios_weekly_rows:
            ios_row.update({k: v for k, v in row.items() if k != "date"})
        for col, val in ios_row.items():
            if col == "date":
                continue
            weekly_df[col] = np.nan
            mask = weekly_df["date"] == today_week
            if mask.any():
                weekly_df.loc[mask, col] = val

    # Ratings count and rank (one value each — current snapshot)
    for app_key in APPS:
        prefix = APP_PREFIX[app_key]
        weekly_df[f"{prefix}_ios_ratings_total"] = np.nan
        weekly_df[f"{prefix}_ios_rank"] = np.nan
        mask = weekly_df["date"] == today_week
        if mask.any():
            weekly_df.loc[mask, f"{prefix}_ios_ratings_total"] = ratings_counts.get(app_key)
            weekly_df.loc[mask, f"{prefix}_ios_rank"] = ranks.get(app_key)

    return weekly_df


# ── Main ───────────────────────────────────────────────────────────────────────

def pull_appstore() -> pd.DataFrame:
    """
    Full pipeline:
      1. Pull GP reviews with text for all 5 apps
      2. VADER sentiment per review
      3. Aggregate to weekly per-app signals
      4. Add cross-app derived features
      5. Attach iTunes current-week snapshot (rank, ratings count, iOS sentiment)
    """
    print("\n" + "=" * 60)
    print("  App Store Pull — Google Play (historical) + iTunes (snapshot)")
    print("  Primary: GP review velocity + VADER sentiment")
    print("  NOTE: foot traffic excluded — wrong causal chain for delivery app")
    print("=" * 60)

    # 1. Pull GP reviews for all apps
    print("\n--- Google Play Review Pull ---")
    all_weekly: list[pd.DataFrame] = []

    for app_key in APPS:
        print(f"\n  [{app_key}] Pulling GP reviews …")
        raw_df = _pull_gp_reviews_for_app(app_key)  # uses MAX_GP_REVIEWS_BY_APP[app_key]

        if raw_df.empty:
            print(f"  [{app_key}] No data — skipping.")
            continue

        date_min = raw_df["date"].min().date()
        date_max = raw_df["date"].max().date()
        print(f"  [{app_key}] {len(raw_df):,} reviews | {date_min} → {date_max}")

        # 2. VADER sentiment
        print(f"  [{app_key}] Computing VADER sentiment …", end=" ", flush=True)
        raw_df = _add_vader_sentiment(raw_df)
        mean_c = raw_df["vader_compound"].mean()
        complaint_pct = raw_df["is_complaint"].mean() * 100
        print(f"mean_compound={mean_c:.3f} | complaint_rate={complaint_pct:.1f}%")

        # 3. Weekly aggregation
        wk = _agg_to_weekly(raw_df, app_key)
        all_weekly.append(wk)

    if not all_weekly:
        raise RuntimeError("All GP pulls returned empty — check network/rate limits.")

    # 4. Outer join all apps on date
    from functools import reduce
    weekly_df = reduce(
        lambda a, b: pd.merge(a, b, on="date", how="outer"),
        all_weekly,
    )
    weekly_df = weekly_df.sort_values("date").reset_index(drop=True)

    print(f"\n  Combined weekly DataFrame: {weekly_df.shape} | "
          f"{weekly_df['date'].min().date()} → {weekly_df['date'].max().date()}")

    # 5. Cross-app derived features
    weekly_df = add_cross_app_features(weekly_df)

    # 6. iTunes snapshots (current week)
    print("\n--- iTunes Snapshot (current week only) ---")
    ratings_counts, ranks = pull_itunes_snapshots()
    print(f"  Ranks: { {APP_PREFIX[k]: v for k, v in ranks.items()} }")
    print(f"  iOS ratings: { {APP_PREFIX[k]: f'{v:,}' if v else 'N/A' for k, v in ratings_counts.items()} }")

    # iTunes RSS reviews for current-week iOS sentiment
    print("  Pulling iTunes RSS reviews for iOS sentiment (current week) …")
    ios_reviews_by_app: dict[str, pd.DataFrame] = {}
    for app_key, meta in APPS.items():
        ios_df = _pull_itunes_ios_reviews(meta["ios"])
        ios_reviews_by_app[app_key] = ios_df
        print(f"    {APP_PREFIX[app_key]}: {len(ios_df)} iOS reviews")

    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    weekly_df = _attach_itunes_snapshot(
        weekly_df, ratings_counts, ranks, ios_reviews_by_app, today_str
    )

    print_pull_summary("App Store Rankings (weekly)", weekly_df, "date")
    return weekly_df


def save_appstore() -> None:
    df = pull_appstore()
    APPSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(APPSTORE_PATH, index=False)
    print(f"Saved → {APPSTORE_PATH}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Shape:   {df.shape}")


if __name__ == "__main__":
    save_appstore()
