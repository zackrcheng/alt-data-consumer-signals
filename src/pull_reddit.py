"""
pull_reddit.py — Reddit sentiment for consumer churn and Dasher supply signals.

Role: Corroborating evidence ONLY — not a model feature. Never pass these columns
to the OLS feature selector (see project spec §18 code standards).
  Consumer signal: rising complaint ratio → churn risk → GOV headwind (4–6 week lead)
  Supply signal:   rising Dasher stress → incentive spend → EBITDA compression
  Deactivation spike: DASH culling low-quality Dashers → short-term margin improvement signal

Historical data strategy (attempted in order):
  1. Arctic Shift API — primary, no auth, back to 2020
  2. Pushshift.io     — fallback, handles 402/403 gracefully (often paywalled)
  3. PRAW             — last resort, ~2–4 weeks of recent posts only

Minimum density: < 50 posts/week → signal_available = False for that subreddit.

Consumer subreddits → data/raw/reddit_consumer.csv:
  doordash, UberEats, instacart, grubhub

Supply subreddits → data/raw/reddit_supply.csv:
  doordash_drivers, UberEatsDrivers, gigworkers

Fails gracefully: if entire pull fails, writes empty CSVs with
reddit_signal_available = False. Does NOT raise exceptions that block the pipeline.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import DATA_RAW, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# ── Output paths ───────────────────────────────────────────────────────────────
REDDIT_CONSUMER_PATH = DATA_RAW / "reddit_consumer.csv"
REDDIT_SUPPLY_PATH   = DATA_RAW / "reddit_supply.csv"

# ── Date range ─────────────────────────────────────────────────────────────────
HISTORY_START      = "2020-12-01"
HISTORY_START_UNIX = int(datetime(2020, 12, 1, tzinfo=timezone.utc).timestamp())

# ── Subreddits ─────────────────────────────────────────────────────────────────
CONSUMER_SUBREDDITS = ["doordash", "UberEats", "instacart", "grubhub"]
SUPPLY_SUBREDDITS   = ["doordash_drivers", "UberEatsDrivers", "gigworkers"]
ALL_SUBREDDITS      = CONSUMER_SUBREDDITS + SUPPLY_SUBREDDITS

MIN_POSTS_PER_WEEK  = 50  # density floor — below this, signal_available = False

# ── API endpoints ──────────────────────────────────────────────────────────────
ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
PUSHSHIFT_URL    = "https://api.pushshift.io/reddit/search/submission/"

# ── Keyword lists (Section 8g) ─────────────────────────────────────────────────
COMPLAINT_KEYWORDS = [
    "late", "never arrived", "missing", "wrong order", "cold",
    "cancelled", "refund", "hours", "support", "useless",
    "terrible", "awful", "scam", "never again", "deleted app",
]
SUPPLY_STRESS_KEYWORDS = [
    "base pay", "hidden tip", "tip baiting", "gas",
    "not worth", "quitting", "slow", "no orders",
    "terrible pay", "unfair",
]
DEACTIVATION_KEYWORDS = [
    "deactivated", "account disabled", "permanent deactivation",
    "wrongfully deactivated", "appeal",
]
SUPPLY_POSITIVE_KEYWORDS = [
    "great week", "busy", "good pay", "bonuses",
    "peak pay", "lots of orders",
]
PROMO_KEYWORDS = [
    "free delivery", "discount", "promo", "coupon", "50% off",
    "promotion", "deal", "offer",
]
DELETED_APP_KEYWORDS = [
    "deleted app", "uninstalled", "deleted doordash", "deleted uber",
]

_vader = SentimentIntensityAnalyzer()

# ── Output column schemas ──────────────────────────────────────────────────────
CONSUMER_COLS = [
    "date", "subreddit", "post_count", "complaint_count", "complaint_ratio",
    "mean_compound_score", "weighted_sentiment", "deleted_app_mentions",
    "promo_mentions", "dash_vs_uber_complaint_ratio",
    "dash_sentiment_momentum_4wk", "reddit_signal_available",
]
SUPPLY_COLS = [
    "date", "subreddit", "post_count", "supply_stress_count",
    "supply_stress_index", "deactivation_mentions", "peak_pay_mentions",
    "driver_quit_mentions", "supply_positive_ratio",
    "driver_supply_stress_4wk", "reddit_signal_available",
]


# ── Timestamp conversion helper ────────────────────────────────────────────────

def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Convert a Series of Unix ints or ISO strings to datetime64[ns]."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


# ── Data source: Arctic Shift ──────────────────────────────────────────────────

def _pull_arctic_shift(subreddit: str) -> tuple[pd.DataFrame, bool]:
    """
    Pull from Arctic Shift API, paginating via timestamp cursor.
    Uses seen-ID deduplication to handle timestamp collisions at page boundaries.
    """
    rows: list[dict] = []
    seen_ids: set[str] = set()
    after  = HISTORY_START
    before = pd.Timestamp.today().strftime("%Y-%m-%dT%H:%M:%SZ")
    page   = 0

    while True:
        try:
            resp = requests.get(
                ARCTIC_SHIFT_URL,
                params={
                    "subreddit": subreddit,
                    "after":     after,
                    "before":    before,
                    "limit":     100,
                    "sort":      "asc",
                },
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            print(f"    Arctic Shift [{subreddit}]: request error — {e}")
            break

        if resp.status_code != 200:
            print(f"    Arctic Shift [{subreddit}]: HTTP {resp.status_code}")
            break

        try:
            posts = resp.json().get("data", [])
        except ValueError:
            print(f"    Arctic Shift [{subreddit}]: invalid JSON")
            break

        if not posts:
            break

        new_count = 0
        last_ts   = None
        for p in posts:
            pid = str(p.get("id", ""))
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            ts = p.get("created_utc")
            rows.append({
                "created_utc":  ts,
                "subreddit":    subreddit,
                "title":        p.get("title", ""),
                "selftext":     (p.get("selftext") or "")[:500],
                "score":        p.get("score", 0),
                "num_comments": p.get("num_comments", 0),
            })
            new_count += 1
            last_ts = ts

        page += 1
        if page % 20 == 0:
            print(f"    Arctic Shift [{subreddit}]: page {page} — {len(rows):,} posts …", flush=True)

        if new_count == 0 or last_ts is None:
            break

        # Advance cursor: convert to ISO string for next request
        if isinstance(last_ts, (int, float)):
            after = pd.Timestamp(last_ts, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            after = str(last_ts)

        time.sleep(0.5)

    if not rows:
        return pd.DataFrame(), False

    df = pd.DataFrame(rows)
    df["created_utc"] = _parse_timestamps(df["created_utc"])
    df = df.dropna(subset=["created_utc"])
    return df, True


# ── Data source: Pushshift ─────────────────────────────────────────────────────

def _pull_pushshift(subreddit: str) -> tuple[pd.DataFrame, bool]:
    """Pushshift fallback. Returns empty + False on 401/402/403 (paywalled)."""
    rows: list[dict] = []
    after_unix = HISTORY_START_UNIX

    while True:
        try:
            resp = requests.get(
                PUSHSHIFT_URL,
                params={
                    "subreddit":  subreddit,
                    "after":      after_unix,
                    "size":       100,
                    "sort":       "asc",
                    "sort_type":  "created_utc",
                },
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            print(f"    Pushshift [{subreddit}]: request error — {e}")
            break

        if resp.status_code in (401, 402, 403):
            print(f"    Pushshift [{subreddit}]: HTTP {resp.status_code} — paywalled/blocked.")
            return pd.DataFrame(), False
        if resp.status_code != 200:
            print(f"    Pushshift [{subreddit}]: HTTP {resp.status_code}")
            break

        try:
            posts = resp.json().get("data", [])
        except ValueError:
            break

        if not posts:
            break

        last_ts = after_unix
        for p in posts:
            ts = p.get("created_utc", 0)
            rows.append({
                "created_utc":  ts,
                "subreddit":    subreddit,
                "title":        p.get("title", ""),
                "selftext":     (p.get("selftext") or "")[:500],
                "score":        p.get("score", 0),
                "num_comments": p.get("num_comments", 0),
            })
            if ts:
                last_ts = max(last_ts, int(ts))

        if last_ts == after_unix:
            break
        after_unix = last_ts
        time.sleep(1.0)

    if not rows:
        return pd.DataFrame(), False

    df = pd.DataFrame(rows)
    df["created_utc"] = _parse_timestamps(df["created_utc"])
    df = df.dropna(subset=["created_utc"])
    return df, True


# ── Data source: PRAW ──────────────────────────────────────────────────────────

def _load_praw_client():
    """Load PRAW from env. Returns (reddit, True) or (None, False)."""
    cid  = os.getenv("REDDIT_CLIENT_ID",     "").strip()
    csec = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        return None, False
    try:
        import praw
        reddit = praw.Reddit(
            client_id=cid,
            client_secret=csec,
            user_agent="dash_research_bot/0.1",
        )
        return reddit, True
    except ImportError:
        print("  praw not installed: pip install praw")
        return None, False
    except Exception as e:
        print(f"  PRAW init failed: {e}")
        return None, False


def _pull_praw(subreddit: str, reddit) -> tuple[pd.DataFrame, bool]:
    """PRAW last resort — returns ~2–4 weeks of recent posts only."""
    rows = []
    try:
        for post in reddit.subreddit(subreddit).new(limit=1000):
            rows.append({
                "created_utc":  datetime.fromtimestamp(post.created_utc, tz=timezone.utc).replace(tzinfo=None),
                "subreddit":    subreddit,
                "title":        post.title or "",
                "selftext":     (post.selftext or "")[:500],
                "score":        post.score,
                "num_comments": post.num_comments,
            })
    except Exception as e:
        print(f"    PRAW [{subreddit}]: error — {e}")
        return pd.DataFrame(), False

    if not rows:
        return pd.DataFrame(), False

    df = pd.DataFrame(rows)
    df["created_utc"] = pd.to_datetime(df["created_utc"])
    return df, True


def _pull_subreddit(subreddit: str, reddit) -> tuple[pd.DataFrame, str]:
    """Try Arctic Shift → Pushshift → PRAW. Returns (df, source_name)."""
    print(f"\n  [{subreddit}] Arctic Shift …", flush=True)
    df, ok = _pull_arctic_shift(subreddit)
    if ok and not df.empty:
        return df, "Arctic Shift"

    print(f"  [{subreddit}] Arctic Shift failed. Pushshift …")
    df, ok = _pull_pushshift(subreddit)
    if ok and not df.empty:
        return df, "Pushshift"

    if reddit is not None:
        print(f"  [{subreddit}] Pushshift failed. PRAW (recent only) …")
        df, ok = _pull_praw(subreddit, reddit)
        if ok and not df.empty:
            return df, "PRAW (recent ~2–4 wks)"

    print(f"  [{subreddit}] All sources failed.")
    return pd.DataFrame(), "None"


# ── Sentiment scoring ──────────────────────────────────────────────────────────

def _score_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER compound + keyword flags to every post.
    Posts with < 10 words get vader_compound=NaN and all flags=0 (too short for VADER).
    Keyword classification is a hard override: complaint keyword present → is_complaint=1
    regardless of VADER score.
    """
    df = df.copy()
    texts      = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()
    word_count = texts.str.split().str.len().fillna(0).astype(int)
    texts_lower = texts.str.lower()

    # VADER only on posts long enough to produce meaningful scores
    long_mask = word_count >= 10
    compounds = np.full(len(df), np.nan)
    if long_mask.any():
        compounds[long_mask] = [
            _vader.polarity_scores(t)["compound"]
            for t in texts[long_mask]
        ]
    df["vader_compound"] = compounds

    def _kw_hit(kw_list: list[str]) -> pd.Series:
        return texts_lower.apply(lambda t: int(any(kw in t for kw in kw_list)))

    complaint_kw = _kw_hit(COMPLAINT_KEYWORDS)

    # Complaint: VADER negative OR keyword present (keyword is hard override)
    df["is_complaint"] = (
        ((df["vader_compound"].fillna(1.0) < -0.2) | complaint_kw.astype(bool))
    ).astype(int)

    df["is_supply_stress"]   = _kw_hit(SUPPLY_STRESS_KEYWORDS)
    df["is_deactivation"]    = _kw_hit(DEACTIVATION_KEYWORDS)
    df["is_supply_positive"] = _kw_hit(SUPPLY_POSITIVE_KEYWORDS)
    df["is_peak_pay"]        = texts_lower.str.contains("peak pay", na=False).astype(int)
    df["is_driver_quit"]     = texts_lower.str.contains(
        r"\b(?:quit|quitting|leaving|left the platform)\b", na=False, regex=True
    ).astype(int)
    df["is_promo"]           = _kw_hit(PROMO_KEYWORDS)
    df["is_deleted_app"]     = _kw_hit(DELETED_APP_KEYWORDS)

    # Zero all flags for short posts
    short_mask = ~long_mask
    flag_cols = [
        "is_complaint", "is_supply_stress", "is_deactivation",
        "is_supply_positive", "is_peak_pay", "is_driver_quit",
        "is_promo", "is_deleted_app",
    ]
    df.loc[short_mask, flag_cols] = 0

    return df


# ── Weekly aggregation ─────────────────────────────────────────────────────────

def _week_end_sunday(ts_series: pd.Series) -> pd.Series:
    """Map timestamp → week-ending Sunday (normalized to midnight)."""
    return ts_series.dt.to_period("W").dt.end_time.dt.normalize()


def _agg_consumer(df: pd.DataFrame, subreddit: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = _week_end_sunday(df["created_utc"])
    grp = df.groupby("date")

    def _weighted_compound(g: pd.DataFrame) -> float:
        w    = (g["score"].fillna(0).clip(lower=0) + 1).values
        vals = g["vader_compound"].fillna(0).values
        return float(np.average(vals, weights=w))

    weekly = pd.DataFrame({
        "post_count":           grp.size(),
        "complaint_count":      grp["is_complaint"].sum(),
        "complaint_ratio":      grp["is_complaint"].mean(),
        "mean_compound_score":  grp["vader_compound"].mean(),
        "weighted_sentiment":   grp.apply(_weighted_compound, include_groups=False),
        "deleted_app_mentions": grp["is_deleted_app"].sum(),
        "promo_mentions":       grp["is_promo"].sum(),
    }).reset_index()

    weekly["subreddit"] = subreddit
    return weekly


def _agg_supply(df: pd.DataFrame, subreddit: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = _week_end_sunday(df["created_utc"])
    grp = df.groupby("date")

    weekly = pd.DataFrame({
        "post_count":            grp.size(),
        "supply_stress_count":   grp["is_supply_stress"].sum(),
        "supply_stress_index":   grp["is_supply_stress"].mean(),
        "deactivation_mentions": grp["is_deactivation"].sum(),
        "peak_pay_mentions":     grp["is_peak_pay"].sum(),
        "driver_quit_mentions":  grp["is_driver_quit"].sum(),
        "supply_positive_ratio": grp["is_supply_positive"].mean(),
    }).reset_index()

    weekly["subreddit"] = subreddit
    return weekly


def _check_density(weekly: pd.DataFrame, subreddit: str) -> bool:
    """Return True if subreddit meets the 50 posts/week minimum."""
    if weekly.empty:
        return False
    mean_ppw = weekly["post_count"].mean()
    if mean_ppw < MIN_POSTS_PER_WEEK:
        print(f"  [{subreddit}] {mean_ppw:.1f} posts/wk < {MIN_POSTS_PER_WEEK} → signal_available=False")
        return False
    return True


# ── Cross-platform derived metrics ─────────────────────────────────────────────

def _add_consumer_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append dash_vs_uber_complaint_ratio (doordash rows only) and
    dash_sentiment_momentum_4wk (doordash rows only).
    All other subreddit rows get NaN for these columns.
    """
    df = df.copy().sort_values(["subreddit", "date"]).reset_index(drop=True)
    df["dash_vs_uber_complaint_ratio"] = np.nan
    df["dash_sentiment_momentum_4wk"]  = np.nan

    # Complaint ratio: doordash / UberEats, aligned by date
    dd = df[df["subreddit"] == "doordash"][["date", "complaint_ratio"]].copy()
    ue = df[df["subreddit"] == "UberEats"][["date", "complaint_ratio"]].copy()
    if not dd.empty and not ue.empty:
        merged = dd.merge(ue, on="date", how="left", suffixes=("_dd", "_ue"))
        merged["ratio"] = merged["complaint_ratio_dd"] / merged["complaint_ratio_ue"].replace(0, np.nan)
        ratio_map = merged.set_index("date")["ratio"]
        dd_mask = df["subreddit"] == "doordash"
        df.loc[dd_mask, "dash_vs_uber_complaint_ratio"] = df.loc[dd_mask, "date"].map(ratio_map).values

    # 4-week rolling mean compound score for r/doordash
    dd_mask = df["subreddit"] == "doordash"
    if dd_mask.any():
        rolling = (
            df.loc[dd_mask, "mean_compound_score"]
            .rolling(4, min_periods=1)
            .mean()
        )
        df.loc[dd_mask, "dash_sentiment_momentum_4wk"] = rolling.values

    return df


def _add_supply_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Append driver_supply_stress_4wk (doordash_drivers rows only)."""
    df = df.copy().sort_values(["subreddit", "date"]).reset_index(drop=True)
    df["driver_supply_stress_4wk"] = np.nan

    dd_mask = df["subreddit"] == "doordash_drivers"
    if dd_mask.any():
        rolling = (
            df.loc[dd_mask, "supply_stress_index"]
            .rolling(4, min_periods=1)
            .mean()
        )
        df.loc[dd_mask, "driver_supply_stress_4wk"] = rolling.values

    return df


# ── Fallback empty CSVs ────────────────────────────────────────────────────────

def _write_fallback_csvs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CONSUMER_COLS).to_csv(REDDIT_CONSUMER_PATH, index=False)
    pd.DataFrame(columns=SUPPLY_COLS).to_csv(REDDIT_SUPPLY_PATH, index=False)
    print("  Wrote empty fallback reddit_consumer.csv and reddit_supply.csv.")
    print("  reddit_signal_available = False for all subreddits.")


# ── Main ───────────────────────────────────────────────────────────────────────

def pull_reddit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline. Returns (consumer_df, supply_df) — both weekly long-format.
    Corroborating evidence only: never pass these columns to the OLS feature selector.
    """
    print("\n" + "=" * 60)
    print("  Reddit Sentiment Pull — Corroborating Evidence Only")
    print("  NOT a model feature: qualitative narrative and risk corroboration")
    print(f"  History target: {HISTORY_START} → present")
    print("=" * 60)

    # Credential status check (Arctic Shift / Pushshift need no credentials)
    cid  = os.getenv("REDDIT_CLIENT_ID",     "").strip()
    csec = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        print("\n  SETUP NOTICE: PRAW credentials not found in .env.")
        print("    Add to .env:  REDDIT_CLIENT_ID=your_id")
        print("                  REDDIT_CLIENT_SECRET=your_secret")
        print("    Register at:  reddit.com/prefs/apps → create 'script' app")
        print("    User agent:   dash_research_bot/0.1")
        print("  Continuing — Arctic Shift and Pushshift require no credentials.\n")

    reddit, praw_ok = _load_praw_client()

    # Pull all subreddits
    raw_data:    dict[str, pd.DataFrame] = {}
    source_used: dict[str, str]          = {}
    for sub in ALL_SUBREDDITS:
        df, source = _pull_subreddit(sub, reddit if praw_ok else None)
        raw_data[sub]    = df
        source_used[sub] = source

    if all(df.empty for df in raw_data.values()):
        print("\n  All pulls returned empty.")
        return pd.DataFrame(), pd.DataFrame()

    # Score sentiment
    scored: dict[str, pd.DataFrame] = {}
    for sub, df in raw_data.items():
        if df.empty:
            scored[sub] = df
            continue
        print(f"\n  [{sub}] Scoring {len(df):,} posts …", end=" ", flush=True)
        scored[sub] = _score_posts(df)
        mean_c = scored[sub]["vader_compound"].dropna().mean()
        cr     = scored[sub]["is_complaint"].mean() * 100
        print(f"mean_compound={mean_c:.3f}  complaint_rate={cr:.1f}%")

    # Consumer aggregation
    consumer_weeks: list[pd.DataFrame] = []
    signal_avail:   dict[str, bool]    = {}

    for sub in CONSUMER_SUBREDDITS:
        df = scored.get(sub, pd.DataFrame())
        if df.empty:
            signal_avail[sub] = False
            continue
        wk    = _agg_consumer(df, sub)
        avail = _check_density(wk, sub)
        signal_avail[sub]          = avail
        wk["reddit_signal_available"] = avail
        consumer_weeks.append(wk)

    # Supply aggregation
    supply_weeks: list[pd.DataFrame] = []

    for sub in SUPPLY_SUBREDDITS:
        df = scored.get(sub, pd.DataFrame())
        if df.empty:
            signal_avail[sub] = False
            continue
        wk    = _agg_supply(df, sub)
        avail = _check_density(wk, sub)
        signal_avail[sub]          = avail
        wk["reddit_signal_available"] = avail
        supply_weeks.append(wk)

    # Build final DataFrames
    consumer_df = (
        pd.concat(consumer_weeks, ignore_index=True)
        if consumer_weeks
        else pd.DataFrame(columns=CONSUMER_COLS)
    )
    supply_df = (
        pd.concat(supply_weeks, ignore_index=True)
        if supply_weeks
        else pd.DataFrame(columns=SUPPLY_COLS)
    )

    if not consumer_df.empty:
        consumer_df = _add_consumer_derived(consumer_df)
    if not supply_df.empty:
        supply_df = _add_supply_derived(supply_df)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REDDIT PULL SUMMARY")
    print("=" * 60)

    for sub in ALL_SUBREDDITS:
        df    = raw_data.get(sub, pd.DataFrame())
        src   = source_used.get(sub, "None")
        avail = signal_avail.get(sub, False)
        if df.empty:
            print(f"  {sub:<26} source=None                 posts=0  signal_available=False")
            continue
        d_min = df["created_utc"].min().date()
        d_max = df["created_utc"].max().date()
        n     = len(df)
        weeks = max((df["created_utc"].max() - df["created_utc"].min()).days / 7, 1)
        print(
            f"  {sub:<26} source={src:<22} posts={n:>7,}  "
            f"{d_min}→{d_max}  {n/weeks:.0f}/wk  signal_available={avail}"
        )

    # Year-by-year complaint trend for r/doordash
    if "doordash" in scored and not scored["doordash"].empty:
        dd = scored["doordash"].copy()
        dd["year"] = dd["created_utc"].dt.year
        print("\n  r/doordash — mean complaint ratio by year:")
        for yr, ratio in dd.groupby("year")["is_complaint"].mean().items():
            print(f"    {yr}: {ratio:.3f}")

    # Current 4-week Dasher supply stress
    if not supply_df.empty and "subreddit" in supply_df.columns:
        dd_sup = (
            supply_df[supply_df["subreddit"] == "doordash_drivers"]
            .sort_values("date")
            .tail(4)
        )
        if not dd_sup.empty and "supply_stress_index" in dd_sup.columns:
            stress_4wk = dd_sup["supply_stress_index"].mean()
            print(f"\n  r/doordash_drivers — current 4-week supply stress index: {stress_4wk:.3f}")

    return consumer_df, supply_df


def save_reddit() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    try:
        consumer_df, supply_df = pull_reddit()
    except Exception as e:
        print(f"\n  Reddit pull failed: {e}")
        print("  Reddit pull failed. Signal set to unavailable. Proceeding with pipeline.")
        _write_fallback_csvs()
        return

    if not consumer_df.empty:
        consumer_df.to_csv(REDDIT_CONSUMER_PATH, index=False)
        print(f"\nSaved → {REDDIT_CONSUMER_PATH}  shape={consumer_df.shape}")
        print(f"Columns ({len(consumer_df.columns)}): {list(consumer_df.columns)}")
    else:
        pd.DataFrame(columns=CONSUMER_COLS).to_csv(REDDIT_CONSUMER_PATH, index=False)
        print(f"\nSaved empty fallback → {REDDIT_CONSUMER_PATH}")

    if not supply_df.empty:
        supply_df.to_csv(REDDIT_SUPPLY_PATH, index=False)
        print(f"Saved → {REDDIT_SUPPLY_PATH}  shape={supply_df.shape}")
        print(f"Columns ({len(supply_df.columns)}): {list(supply_df.columns)}")
    else:
        pd.DataFrame(columns=SUPPLY_COLS).to_csv(REDDIT_SUPPLY_PATH, index=False)
        print(f"Saved empty fallback → {REDDIT_SUPPLY_PATH}")


if __name__ == "__main__":
    save_reddit()
