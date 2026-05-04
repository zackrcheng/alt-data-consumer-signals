"""
pull_job_postings.py — Job posting data for DASH/UBER/CART (corroborating evidence).

Role: Corroborating evidence ONLY. Never pass these columns to the OLS feature
selector (see CLAUDE.md §18 code standards).

Investment logic (CLAUDE.md §8h):
  US merchant sales hiring     → near-term GOV expansion in new US markets
  International/Deliveroo      → Q2-Q3 2026 international GOV acceleration thesis
  DASH vs. UBER hiring ratio   → relative competitive intensity signal

Sources attempted (in order per spec):
  1. Indeed HTML scraping (primary)  — 2s sleep, max 3 pages per query
  2. Greenhouse JSON API             — DoorDash careers spot check per spec;
                                       also used as peer fallback when Indeed blocks

LinkedIn: excluded (ToS violation per CLAUDE.md §8h).

Source-by-source status (verified 2026-05-04):
  - Indeed: HTTP 403 across all queries (bot detection on free tier).
    Kept in pipeline since 403s are quick and graceful, but does not contribute
    data without paid proxy infra (BrightData / ScraperAPI).
  - DoorDash Greenhouse token "doordashusa" → 200 OK, ~455 jobs. (The naive
    "doordash" token 404s; the correct board lives at job-boards.greenhouse.io/
    doordashusa.)
  - Uber custom careers API: POST www.uber.com/api/loadSearchJobsResults with
    body {"params":{"limit":N,"page":P}} and any x-csrf-token header. Returns
    ~1066 open jobs. This is the same endpoint jobs.uber.com/uber.com/careers
    use under the hood.
  - Instacart Greenhouse token "instacart" → 200 OK.

Deduplication: hash(title.lower() + company.lower() + city.lower()) within ±7 days.
Keeps only first occurrence per hash within each 7-day window.

Fails gracefully: if all scraping fails writes an empty CSV with signal_available=False
and prints "Job postings pull failed. Using as Future Work only." Never raises
exceptions that block the main pipeline.
"""

import hashlib
import re
import time
import urllib.parse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import DATA_RAW, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# ── Output path ────────────────────────────────────────────────────────────────
JOB_POSTINGS_PATH = DATA_RAW / "job_postings.csv"

# ── Scrape parameters ──────────────────────────────────────────────────────────
SCRAPE_DATE     = pd.Timestamp.today().normalize()
MAX_PAGES       = 3      # 3 pages per query (spec: max 3 pages / ~30 results per query)
SLEEP_SEC       = 2.0    # between Indeed requests
REQUEST_TIMEOUT = 20     # seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Company configuration ──────────────────────────────────────────────────────
COMPANY_CONFIG = {
    "DASH": {
        "display_name":       "DoorDash",
        "indeed_term":        "DoorDash",
        # Eight Greenhouse boards across the DoorDash corporate family (all
        # verified by direct probe, 2026-05-04). DoorDash's international
        # footprint spans:
        #   doordashusa           — US delivery (~455 jobs)         [core delivery]
        #   doordashinternational — Misc intl G&A (~18 jobs)         [core delivery]
        #   doordashcanada        — Canadian delivery (~26 jobs)     [core delivery]
        #   doordashaustralia     — Australian delivery (~29 jobs)   [core delivery]
        #                            Replaced Deliveroo AU after their 2022 exit
        #   doordashmexico        — LATAM delivery (~40 jobs)        [core delivery]
        #   wolt                  — European delivery (~254 jobs)    [core delivery]
        #                            Acquired 2022; Finland HQ, ~30 European countries
        #   sevenroomsuk          — B2B restaurant SaaS (~12 jobs)   [non-delivery
        #                            subsidiary; excluded from headline ratio]
        #   bbot                  — Restaurant ordering tech (~1 job) [non-delivery
        #                            subsidiary; excluded from headline ratio]
        # Deliveroo (acquired) uses Ashby — handled by _pull_ashby_deliveroo().
        "greenhouse_tokens":  [
            "doordashusa", "doordashinternational", "doordashcanada",
            "doordashaustralia", "doordashmexico",
            "wolt", "sevenroomsuk", "bbot",
        ],
        "run_dasher_queries": True,   # Dasher supply roles: DASH only
        "run_intl_queries":   True,   # Deliveroo-market hiring: DASH only
    },
    "UBER": {
        "display_name":       "Uber",
        "indeed_term":        "Uber Eats",
        "greenhouse_tokens":  [],     # Uber uses its own careers API, not Greenhouse
        "run_dasher_queries": False,
        "run_intl_queries":   False,
    },
    "CART": {
        "display_name":       "Instacart",
        "indeed_term":        "Instacart",
        "greenhouse_tokens":  ["instacart"],
        "run_dasher_queries": False,
        "run_intl_queries":   False,
    },
}

# ── Query sets ─────────────────────────────────────────────────────────────────
US_ROLE_QUERIES = [
    "merchant account executive",
    "merchant sales representative",
    "regional operations",
]
DASH_ONLY_QUERIES = [
    "dasher acquisition",
    "dasher supply growth",
]
INTL_ROLE_QUERIES = [
    "merchant account executive",
    "merchant sales",
]

# Deliveroo markets → country-specific Indeed subdomains
INTL_DOMAINS = {
    "United Kingdom": "uk",
    "Germany":        "de",
    "France":         "fr",
    "Australia":      "au",
}

# ── Classification constants ───────────────────────────────────────────────────
# Tier-1 cities are excluded from is_ops_expansion (spec: "not NYC/LA/SF/CHI/BOS")
TIER1_CITIES = frozenset([
    "new york", "new york city", "nyc",
    "los angeles", "la",
    "san francisco", "sf", "san francisco bay area",
    "chicago",
    "boston",
])

MERCHANT_SALES_KWS = [
    "merchant", "account executive", "sales representative",
    "business development", "account manager", "sales manager",
    "account lead", "sales lead", "merchant growth",
    "partner success", "merchant success",
]
OPS_EXPANSION_KWS = [
    "operations", "regional", "market manager", "city manager",
    "market lead", "ops manager", "operations manager",
    "regional manager", "market expansion",
]
DASHER_SUPPLY_KWS = [
    "dasher", "driver", "courier", "supply growth",
    "supply acquisition", "fleet", "gig worker",
]

# Uber-only business-line filter: required for apples-to-apples vs. DASH.
# Uber's API organizes by function (Sales/Eng/Ops), not by business line, so a
# generic "Sales Manager" could be Mobility, Freight, ATG, or Eats. We keep only
# titles that explicitly indicate the delivery business (Uber Eats, Grocery,
# Postmates legacy). DASH and CART are pure-delivery companies — no equivalent
# filter needed.
UBER_DELIVERY_LINE_KWS = [
    "eats", "delivery", "deliveries", "grocery", "groceries",
    "restaurant", "merchant", "courier", "food", "postmates",
    "shopper", "convenience", "alcohol", "retail",
]
# Negative filter: skip Uber jobs whose title is unambiguously another business line
UBER_OTHER_LINE_KWS = [
    "mobility", "rides", "rider", "driver-partner",
    "freight", "trucking", "logistics platform",
    "atg", "advanced technologies", "elevate", "uber for business",
    "transit", "rentals", "hourly",
]

DELIVEROO_COUNTRIES = frozenset(["United Kingdom", "Germany", "France", "Australia"])
US_COUNTRY_ALIASES  = frozenset(["United States", "US", "USA"])

# Sources tagged as "non-delivery" (corporate subsidiaries that aren't part of
# the consumer-delivery business line). Excluded from the apples-to-apples
# DASH-vs-UBER-Delivery ratio but kept in the dataset for transparency.
NON_DELIVERY_SOURCES = frozenset([
    "greenhouse_sevenroomsuk",   # B2B restaurant CRM SaaS
    "greenhouse_bbot",           # B2B restaurant ordering tech
])

# ── Output columns ─────────────────────────────────────────────────────────────
OUTPUT_COLS = [
    "date_scraped", "date_posted", "company", "job_title",
    "location_city", "location_state", "location_country",
    "is_merchant_sales", "is_ops_expansion", "is_dasher_supply",
    "is_international", "is_deliveroo_market",
    "source", "signal_available",
]


# ── Date parsing ───────────────────────────────────────────────────────────────

def _parse_relative_date(raw: str) -> pd.Timestamp:
    """Convert Indeed relative date strings to absolute Timestamp."""
    s = raw.strip().lower()
    if not s or s in ("just posted", "today", "active today", "new"):
        return SCRAPE_DATE
    m = re.search(r"(\d+)\+?\s+hour", s)
    if m:
        return SCRAPE_DATE
    m = re.search(r"(\d+)\+?\s+day", s)
    if m:
        days = min(int(m.group(1)), 60)  # cap at 60 days
        return SCRAPE_DATE - pd.Timedelta(days=days)
    m = re.search(r"(\d+)\+?\s+month", s)
    if m:
        return SCRAPE_DATE - pd.Timedelta(days=30 * int(m.group(1)))
    return SCRAPE_DATE


# ── Location parsing ───────────────────────────────────────────────────────────

_US_STATE_ABBREVS = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
}

_COUNTRY_FROM_DOMAIN = {
    "uk": "United Kingdom",
    "de": "Germany",
    "fr": "France",
    "au": "Australia",
    "ca": "Canada",
}

# Map common country name variations to canonical forms
_COUNTRY_ALIASES = {
    "england":           "United Kingdom",
    "scotland":          "United Kingdom",
    "wales":             "United Kingdom",
    "uk":                "United Kingdom",
    "gb":                "United Kingdom",
    "great britain":     "United Kingdom",
    "deutschland":       "Germany",
    "france":            "France",
    "australia":         "Australia",
    # Australian state/territory abbreviations
    "nsw":               "Australia",
    "vic":               "Australia",
    "qld":               "Australia",
    "sa":                "Australia",
    "wa":                "Australia",
    "tas":               "Australia",
    "act":               "Australia",
    "nt":                "Australia",
    "new south wales":   "Australia",
    "victoria":          "Australia",
    "queensland":        "Australia",
    "western australia": "Australia",
    "us":                "United States",
    "usa":               "United States",
    "united states":     "United States",
}


def _parse_location(raw: str, domain: str = "www") -> tuple[str, str, str]:
    """
    Parse a single 'City, STATE' or 'City, Country' or 'Remote' into
    (city, state, country). For multi-location strings (semicolon-separated
    like 'Berlin, Germany; Helsinki, Finland'), use _parse_locations() instead.

    domain: Indeed country subdomain used for this query (e.g. 'uk', 'de').
            Greenhouse calls pass domain='www' and rely on the country name
            inside the location string itself.
    """
    raw = raw.strip()
    if not raw or raw.lower() in ("remote", "work from home", "anywhere"):
        country = _COUNTRY_FROM_DOMAIN.get(domain, "")
        return "Remote", "", country

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        city = parts[0]
        state = ""
        # Single token with no country hint — only assume US for Indeed's www
        # domain (which IS US-specific). For Greenhouse where domain='www' is
        # just the default placeholder, leave country empty.
        country = _COUNTRY_FROM_DOMAIN.get(domain, "")
        return city, state, country

    city = parts[0]
    second = parts[1].strip()

    # Two-letter US state abbreviation → US
    if second.upper() in _US_STATE_ABBREVS:
        return city, second.upper(), "United States"

    # Known country alias (handles 'England', 'NSW', 'Deutschland' etc.)
    canon = _COUNTRY_ALIASES.get(second.lower())
    if canon:
        return city, "", canon

    # Otherwise the second token IS the country name as written
    # (e.g. 'Berlin, Germany' → country='Germany').
    return city, "", second


def _parse_locations(raw: str, domain: str = "www") -> list[tuple[str, str, str]]:
    """
    Parse a possibly multi-location string into a list of (city, state, country)
    tuples. Wolt and other Greenhouse boards return semicolon-separated lists
    like 'Berlin, Germany; Helsinki, Finland; Tallinn, Estonia' for hybrid roles.
    """
    raw = (raw or "").strip()
    if not raw:
        return [("", "", "")]
    pieces = [p.strip() for p in raw.split(";") if p.strip()]
    return [_parse_location(p, domain) for p in pieces] or [_parse_location(raw, domain)]


# ── Job classification ─────────────────────────────────────────────────────────

def _classify(title: str, city: str, country: str) -> dict:
    """Return boolean classification flags for a single job posting."""
    t = title.lower()
    c = city.lower()

    is_merchant_sales = int(any(kw in t for kw in MERCHANT_SALES_KWS))
    is_ops_expansion  = int(
        any(kw in t for kw in OPS_EXPANSION_KWS) and c not in TIER1_CITIES
    )
    is_dasher_supply  = int(any(kw in t for kw in DASHER_SUPPLY_KWS))
    is_international  = int(country not in US_COUNTRY_ALIASES and bool(country))
    is_deliveroo      = int(country in DELIVEROO_COUNTRIES)

    return {
        "is_merchant_sales":   is_merchant_sales,
        "is_ops_expansion":    is_ops_expansion,
        "is_dasher_supply":    is_dasher_supply,
        "is_international":    is_international,
        "is_deliveroo_market": is_deliveroo,
    }


# ── Indeed scraping ────────────────────────────────────────────────────────────

def _build_indeed_url(query: str, company_term: str, domain: str, page: int) -> str:
    """Build paginated Indeed search URL. page is 0-indexed."""
    base = f"https://{domain}.indeed.com/jobs" if domain != "www" else "https://www.indeed.com/jobs"
    q = urllib.parse.quote_plus(f"{company_term} {query}")
    return f"{base}?q={q}&sort=date&fromage=30&start={page * 10}"


def _parse_indeed_page(html: str, domain: str) -> list[dict]:
    """
    Extract job listings from Indeed HTML. Tries multiple element patterns
    to handle Indeed's frequently-changing DOM structure.
    Returns list of raw dicts or [] on failure/bot-detection.
    """
    soup = BeautifulSoup(html, "lxml")

    # Bot-detection heuristic: CAPTCHA pages are short and title-less
    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True).lower() if title_tag else ""
    if "captcha" in title_text or "verify" in title_text:
        print("    Indeed: CAPTCHA detected — skipping page.", flush=True)
        return []

    rows: list[dict] = []

    # Pattern A: data-testid attributes (2024-2025 Indeed structure)
    job_cards = soup.find_all("div", attrs={"data-testid": "slider_item"})
    if not job_cards:
        job_cards = soup.find_all("div", class_=re.compile(r"job_seen_beacon|tapItem"))
    if not job_cards:
        # Pattern B: any <li> or <div> with a data-jk attribute (job key)
        job_cards = soup.find_all(lambda tag: tag.has_attr("data-jk"))

    for card in job_cards:
        # Title — try data-testid first, then class patterns
        title_el = (
            card.find(attrs={"data-testid": "jobTitle"})
            or card.find("h2", class_=re.compile(r"jobTitle|job-title", re.I))
            or card.find("a", attrs={"data-jk": True})
        )
        if not title_el:
            continue
        title = title_el.get_text(separator=" ", strip=True)

        # Company
        comp_el = (
            card.find(attrs={"data-testid": "company-name"})
            or card.find(class_=re.compile(r"companyName|company", re.I))
        )
        company_raw = comp_el.get_text(strip=True) if comp_el else ""

        # Location
        loc_el = (
            card.find(attrs={"data-testid": "text-location"})
            or card.find(class_=re.compile(r"companyLocation|job-location", re.I))
        )
        location_raw = loc_el.get_text(strip=True) if loc_el else ""

        # Date
        date_el = (
            card.find(attrs={"data-testid": "myJobsStateDate"})
            or card.find(class_=re.compile(r"\bdate\b", re.I))
            or card.find("span", string=re.compile(r"ago|today|posted|just", re.I))
        )
        date_raw = date_el.get_text(strip=True) if date_el else "Today"

        if not title:
            continue

        city, state, country = _parse_location(location_raw, domain)
        rows.append({
            "title":       title,
            "company_raw": company_raw,
            "date_raw":    date_raw,
            "city":        city,
            "state":       state,
            "country":     country,
        })

    return rows


def _scrape_indeed_query(
    query: str,
    company_key: str,
    domain: str = "www",
) -> list[dict]:
    """
    Scrape up to MAX_PAGES of Indeed results for one (query, company, domain) combo.
    Returns a list of normalized job dicts. Empty list on failure.
    """
    cfg  = COMPANY_CONFIG[company_key]
    term = cfg["indeed_term"]
    name = cfg["display_name"]
    results: list[dict] = []

    for page in range(MAX_PAGES):
        url = _build_indeed_url(query, term, domain, page)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f"    Indeed [{name} | {query} | p{page+1}]: request error — {e}")
            break

        if resp.status_code != 200:
            print(f"    Indeed [{name} | {query} | p{page+1}]: HTTP {resp.status_code}")
            break

        page_rows = _parse_indeed_page(resp.text, domain)
        if not page_rows:
            break  # no results or bot-detected → stop paginating

        for r in page_rows:
            city, state, country = r["city"], r["state"], r["country"]
            flags = _classify(r["title"], city, country)
            results.append({
                "date_scraped":      SCRAPE_DATE,
                "date_posted":       _parse_relative_date(r["date_raw"]),
                "company":           name,
                "job_title":         r["title"],
                "location_city":     city,
                "location_state":    state,
                "location_country":  country,
                "source":            f"indeed_{domain}",
                **flags,
            })

        time.sleep(SLEEP_SEC)

    return results


def pull_indeed_all() -> list[dict]:
    """
    Run all Indeed queries for all companies. Returns raw list of job dicts.
    Handles per-query failures gracefully without stopping the pipeline.
    """
    all_rows: list[dict] = []

    for company_key, cfg in COMPANY_CONFIG.items():
        name = cfg["display_name"]
        print(f"\n  [Indeed | {name}] US role queries …")

        for q in US_ROLE_QUERIES:
            print(f"    Query: '{q}' …", end=" ", flush=True)
            rows = _scrape_indeed_query(q, company_key, domain="www")
            print(f"{len(rows)} results")
            all_rows.extend(rows)

        if cfg["run_dasher_queries"]:
            for q in DASH_ONLY_QUERIES:
                print(f"    Query: '{q}' (DASH-only) …", end=" ", flush=True)
                rows = _scrape_indeed_query(q, company_key, domain="www")
                print(f"{len(rows)} results")
                all_rows.extend(rows)

        if cfg["run_intl_queries"]:
            print(f"  [Indeed | {name}] International / Deliveroo-market queries …")
            for country, subdomain in INTL_DOMAINS.items():
                for q in INTL_ROLE_QUERIES:
                    print(f"    Query: '{q}' | {country} …", end=" ", flush=True)
                    rows = _scrape_indeed_query(q, company_key, domain=subdomain)
                    print(f"{len(rows)} results")
                    all_rows.extend(rows)

    return all_rows


# ── Greenhouse API (DoorDash careers spot check + peer fallback) ───────────────

# ── Greenhouse API ─────────────────────────────────────────────────────────────
# Verified board tokens (2026-05-04):
#   DoorDash   "doordashusa" → 200 OK, ~455 jobs
#   Instacart  "instacart"   → 200 OK
# Uber is handled separately via _pull_uber_careers() — they don't use Greenhouse.

_GREENHOUSE_URL = "https://api.greenhouse.io/v1/boards/{token}/jobs"


def _fetch_greenhouse(board_token: str) -> list[dict]:
    """Fetch all open jobs from a Greenhouse board. Returns raw JSON list."""
    url = _GREENHOUSE_URL.format(token=board_token)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("jobs", [])
        print(f"    Greenhouse [{board_token}]: HTTP {resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"    Greenhouse [{board_token}]: {e}")
    return []


def _pull_greenhouse_company(company_key: str) -> list[dict]:
    """
    Pull jobs from one or more Greenhouse boards for one company.
    Some companies (DASH) split their postings across multiple regional boards
    (US / International / Canada) — all are pulled and tagged by source.
    Filters to relevant role types via keyword matching — Greenhouse returns all
    open roles (engineering, finance, etc.) so keyword filtering is mandatory.
    """
    cfg    = COMPANY_CONFIG[company_key]
    name   = cfg["display_name"]
    tokens = cfg.get("greenhouse_tokens", [])
    if not tokens:
        return []  # company doesn't use Greenhouse (e.g. UBER)

    results: list[dict] = []
    all_kws = MERCHANT_SALES_KWS + OPS_EXPANSION_KWS + DASHER_SUPPLY_KWS

    for token in tokens:
        raw_jobs = _fetch_greenhouse(token)
        if not raw_jobs:
            print(f"    Greenhouse [{name} / {token}]: no data.")
            continue

        kept = 0
        for j in raw_jobs:
            title = j.get("title", "").strip()
            if not title:
                continue
            if not any(kw in title.lower() for kw in all_kws):
                continue

            loc_name = (j.get("location") or {}).get("name", "") or ""

            # updated_at is the best available proxy for date_posted in Greenhouse
            updated = j.get("updated_at", "")
            try:
                date_posted = pd.to_datetime(updated, utc=True).tz_localize(None).normalize()
            except Exception:
                date_posted = SCRAPE_DATE

            # Wolt and other regional boards return semicolon-separated multi-
            # location strings — emit one row per parsed location.
            for city, state, country in _parse_locations(loc_name):
                flags = _classify(title, city, country)
                results.append({
                    "date_scraped":      SCRAPE_DATE,
                    "date_posted":       date_posted,
                    "company":           name,
                    "job_title":         title,
                    "location_city":     city,
                    "location_state":    state,
                    "location_country":  country,
                    "source":            f"greenhouse_{token}",
                    **flags,
                })
                kept += 1
        print(f"    Greenhouse [{name} / {token}]: {len(raw_jobs)} total → {kept} relevant.")
        time.sleep(0.5)

    return results


def pull_greenhouse_all() -> list[dict]:
    """
    Pull Greenhouse jobs for all companies that use Greenhouse (DASH + CART).
    UBER skipped — they don't use Greenhouse; handled by _pull_uber_careers().
    """
    all_rows: list[dict] = []
    for company_key in ("DASH", "CART"):
        cfg  = COMPANY_CONFIG[company_key]
        name = cfg["display_name"]
        print(f"\n  [Greenhouse | {name}] Pulling careers API …", end=" ", flush=True)
        rows = _pull_greenhouse_company(company_key)
        print(f"{len(rows)} relevant postings")
        all_rows.extend(rows)
        time.sleep(1.0)
    return all_rows


# ── Ashby API — Deliveroo ──────────────────────────────────────────────────────
# Deliveroo uses Ashby (not Greenhouse). Their public job board endpoint is
#   https://api.ashbyhq.com/posting-api/job-board/deliveroo
# Returns rich structured location data (addressCountry/addressLocality) so
# no string parsing is needed — much cleaner than Greenhouse for international.

_ASHBY_DELIVEROO_URL = (
    "https://api.ashbyhq.com/posting-api/job-board/deliveroo"
    "?includeCompensation=false"
)


def _pull_ashby_deliveroo() -> list[dict]:
    """
    Pull Deliveroo's open roles from Ashby and tag them as DoorDash subsidiary
    (Deliveroo was acquired by DoorDash). Filters to relevant role types.
    Emits one row per (job × location) for multi-location roles.
    """
    try:
        resp = requests.get(_ASHBY_DELIVEROO_URL, timeout=REQUEST_TIMEOUT,
                            headers={**HEADERS, "Accept": "application/json"})
    except requests.exceptions.RequestException as e:
        print(f"    Ashby [deliveroo]: {e}")
        return []

    if resp.status_code != 200:
        print(f"    Ashby [deliveroo]: HTTP {resp.status_code}")
        return []

    try:
        jobs = resp.json().get("jobs", []) or []
    except ValueError:
        print("    Ashby [deliveroo]: invalid JSON")
        return []

    all_kws = MERCHANT_SALES_KWS + OPS_EXPANSION_KWS + DASHER_SUPPLY_KWS
    results: list[dict] = []
    kept = 0

    for j in jobs:
        title = (j.get("title") or "").strip()
        if not title:
            continue
        if not any(kw in title.lower() for kw in all_kws):
            continue

        try:
            date_posted = (
                pd.to_datetime(j.get("publishedAt"), utc=True)
                .tz_localize(None).normalize()
            )
        except Exception:
            date_posted = SCRAPE_DATE

        # Build the (city, state, country) list from primary + secondary locations
        locations: list[tuple[str, str, str]] = []
        primary = j.get("address", {}).get("postalAddress", {}) or {}
        if primary:
            locations.append((
                (primary.get("addressLocality") or "").strip(),
                (primary.get("addressRegion") or "").strip(),
                (primary.get("addressCountry") or "").strip(),
            ))
        for sec in (j.get("secondaryLocations") or []):
            sec_addr = (sec.get("address") or {}).get("postalAddress", {}) or {}
            if sec_addr:
                locations.append((
                    (sec_addr.get("addressLocality") or "").strip(),
                    (sec_addr.get("addressRegion") or "").strip(),
                    (sec_addr.get("addressCountry") or "").strip(),
                ))
        # If no structured address, fall back to the location string
        if not locations:
            for parsed in _parse_locations(j.get("location") or ""):
                locations.append(parsed)

        for city, state, country in locations:
            flags = _classify(title, city, country)
            results.append({
                "date_scraped":      SCRAPE_DATE,
                "date_posted":       date_posted,
                "company":           "DoorDash",   # Deliveroo = DASH subsidiary
                "job_title":         title,
                "location_city":     city,
                "location_state":    state,
                "location_country":  country,
                "source":            "ashby_deliveroo",
                **flags,
            })
            kept += 1

    print(f"    Ashby [deliveroo]: {len(jobs)} total → {kept} relevant rows.")
    return results


# ── Uber careers API ──────────────────────────────────────────────────────────
# Uber doesn't use Greenhouse. The same JSON endpoint that powers
# www.uber.com/careers and jobs.uber.com is publicly callable:
#   POST https://www.uber.com/api/loadSearchJobsResults?localeCode=en
#   Body: {"params": {"limit": N, "page": P}}
#   Headers: Content-Type: application/json + any non-empty x-csrf-token
# Returns {"data": {"results": [...], "totalResults": {"low": N}}} with rich
# location data: {country, countryName, region, city} per job.

_UBER_URL        = "https://www.uber.com/api/loadSearchJobsResults?localeCode=en"
_UBER_PAGE_LIMIT = 100   # max results per page
_UBER_MAX_PAGES  = 20    # cap at 20 pages × 100 = 2000 (more than total job count)


def _is_uber_delivery_role(title: str) -> bool:
    """
    True iff this Uber title appears to belong to the Delivery / Eats business
    line (apples-to-apples with DASH and CART). Excludes Mobility, Freight, ATG,
    and Uber-for-Business roles.

    Logic:
      - Reject if title contains an unambiguous other-business-line keyword
        (e.g. 'Mobility Operations Manager' → Mobility).
      - Otherwise accept if title contains a delivery-line keyword
        (e.g. 'Account Executive, Uber Eats Restaurant Partnerships' → Eats).
    """
    t = title.lower()
    if any(kw in t for kw in UBER_OTHER_LINE_KWS):
        return False
    return any(kw in t for kw in UBER_DELIVERY_LINE_KWS)


def _pull_uber_careers() -> list[dict]:
    """
    Pull all open Uber roles via their public careers API. Two-stage filter:
      1. Role-type match (merchant sales / ops expansion / dasher supply)
      2. Delivery business-line match (apples-to-apples with DASH/CART)
    Emits one row per (job × location) since Uber lists multi-location roles.
    """
    all_kws = MERCHANT_SALES_KWS + OPS_EXPANSION_KWS + DASHER_SUPPLY_KWS
    results: list[dict] = []
    n_role_match    = 0   # passed role-type filter
    n_delivery_only = 0   # passed both filters → kept

    for page in range(_UBER_MAX_PAGES):
        try:
            resp = requests.post(
                _UBER_URL,
                json={"params": {"limit": _UBER_PAGE_LIMIT, "page": page}},
                headers={
                    **HEADERS,
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                    "x-csrf-token":  "x",   # required header; value not validated
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            print(f"    Uber [page={page}]: request error — {e}")
            break

        if resp.status_code != 200:
            print(f"    Uber [page={page}]: HTTP {resp.status_code}")
            break

        try:
            page_jobs = resp.json().get("data", {}).get("results", []) or []
        except ValueError:
            print(f"    Uber [page={page}]: invalid JSON")
            break

        if not page_jobs:
            break

        for j in page_jobs:
            title = (j.get("title") or "").strip()
            if not title:
                continue

            # Stage 1: role-type filter (merchant sales / ops / dasher supply)
            if not any(kw in title.lower() for kw in all_kws):
                continue
            n_role_match += 1

            # Stage 2: business-line filter (Delivery only, drop Mobility/Freight)
            if not _is_uber_delivery_role(title):
                continue
            n_delivery_only += 1

            # Pre-parsed date — Uber returns ISO strings
            try:
                date_posted = (
                    pd.to_datetime(j.get("creationDate"), utc=True)
                    .tz_localize(None)
                    .normalize()
                )
            except Exception:
                date_posted = SCRAPE_DATE

            # Emit one row per location (Uber lists multi-location roles)
            locations = j.get("allLocations") or [j.get("location")] or []
            for loc in locations:
                if not loc:
                    continue
                city    = (loc.get("city") or "").strip() or "Unspecified"
                state   = (loc.get("region") or "").strip()
                country = (loc.get("countryName") or "").strip()

                flags = _classify(title, city, country)
                results.append({
                    "date_scraped":      SCRAPE_DATE,
                    "date_posted":       date_posted,
                    "company":           "Uber",
                    "job_title":         title,
                    "location_city":     city,
                    "location_state":    state,
                    "location_country":  country,
                    "source":            "uber_careers_api",
                    **flags,
                })

        if len(page_jobs) < _UBER_PAGE_LIMIT:
            break  # last page
        time.sleep(0.5)

    print(f"    Uber filter funnel: {n_role_match} matched role keywords → "
          f"{n_delivery_only} also matched delivery business line")
    return results


# ── Deduplication ──────────────────────────────────────────────────────────────

def _job_hash(title: str, company: str, city: str) -> str:
    """Stable 8-char hash for deduplication key."""
    key = f"{title.lower().strip()}|{company.lower().strip()}|{city.lower().strip()}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove reposted duplicates. Keeps only first occurrence of each
    (title, company, city) hash within a rolling 7-day window.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_hash"] = df.apply(
        lambda r: _job_hash(r["job_title"], r["company"], r["location_city"]),
        axis=1,
    )
    df["date_posted"] = pd.to_datetime(df["date_posted"])
    df = df.sort_values("date_posted").reset_index(drop=True)

    keep = np.ones(len(df), dtype=bool)

    for h in df["_hash"].unique():
        idx   = df.index[df["_hash"] == h].tolist()
        if len(idx) <= 1:
            continue
        dates  = df.loc[idx, "date_posted"].dt.normalize().tolist()
        anchor = dates[0]
        for i, (row_idx, dt) in enumerate(zip(idx[1:], dates[1:]), start=1):
            if abs((dt - anchor).days) <= 7:
                keep[row_idx] = False   # duplicate repost — drop
            else:
                anchor = dt             # new anchor for next 7-day window

    n_dupes = int((~keep).sum())
    print(f"  Deduplication: {n_dupes} duplicates removed from {len(df)} raw postings.")
    return df[keep].drop(columns=["_hash"]).reset_index(drop=True)


# ── Summary statistics & write-up paragraph ────────────────────────────────────

def _compute_summary(df: pd.DataFrame) -> dict:
    """Compute and return the summary statistics dict."""
    def _count(mask_company: pd.Series, *flag_masks) -> int:
        m = mask_company.copy()
        for fm in flag_masks:
            m = m & fm
        return int(m.sum())

    # Apples-to-apples filter: drop non-delivery subsidiaries (e.g. SevenRooms)
    # before computing the headline DASH-vs-UBER-Delivery comparison.
    delivery_only = ~df["source"].isin(NON_DELIVERY_SOURCES)

    dash      = df["company"] == "DoorDash"
    dash_core = dash & delivery_only          # DASH delivery only (no SevenRooms)
    uber      = df["company"] == "Uber"       # already filtered to Delivery business
    cart      = df["company"] == "Instacart"

    ms_us   = df["is_merchant_sales"].astype(bool) & ~df["is_international"].astype(bool)
    ms_intl = df["is_merchant_sales"].astype(bool) &  df["is_international"].astype(bool)

    d_ms_us       = _count(dash_core, ms_us)
    d_ms_intl     = _count(dash_core, ms_intl)
    d_deliveroo   = _count(dash_core, df["is_deliveroo_market"].astype(bool))
    d_dasher      = _count(dash_core, df["is_dasher_supply"].astype(bool))
    d_total_all   = int(dash.sum())               # all DASH-family rows
    d_total_core  = int(dash_core.sum())          # delivery-only

    u_ms_us       = _count(uber, ms_us)
    u_ms_intl     = _count(uber, ms_intl)
    u_total       = int(uber.sum())

    c_total       = int(cart.sum())

    ratio_uber    = round(d_total_core / u_total, 2) if u_total else float("nan")
    ratio_cart    = round(d_total_core / c_total, 2) if c_total else float("nan")

    # International share % — computed against delivery-only DASH base
    d_intl_share = (d_ms_intl / d_total_core * 100) if d_total_core else 0.0
    u_intl_share = (u_ms_intl / u_total       * 100) if u_total      else 0.0
    intl_ratio   = round(d_intl_share / u_intl_share, 1) if u_intl_share else float("nan")

    return {
        "dash_merchant_sales_us_total":    d_ms_us,
        "dash_merchant_sales_intl_total":  d_ms_intl,
        "dash_deliveroo_market_total":     d_deliveroo,
        "dash_dasher_supply_total":        d_dasher,
        "dash_total":                      d_total_core,    # delivery-only headline
        "dash_total_incl_subsidiaries":    d_total_all,     # incl. SevenRooms etc.
        "uber_total":                      u_total,
        "cart_total":                      c_total,
        "dash_vs_uber_posting_ratio":      ratio_uber,
        "dash_vs_cart_posting_ratio":      ratio_cart,
        "dash_intl_share_pct":             round(d_intl_share, 1),
        "uber_intl_share_pct":             round(u_intl_share, 1),
        "intl_ratio_vs_uber":              intl_ratio,
        "uber_merchant_sales_us_total":    u_ms_us,
    }


def _fmt(val, fmt: str = "{:.2f}", na: str = "N/A") -> str:
    """Format a number, returning na if value is NaN/None."""
    if val is None:
        return na
    if isinstance(val, float) and np.isnan(val):
        return na
    return fmt.format(val)


def _print_summary(stats: dict, df: pd.DataFrame) -> None:
    """Print formatted summary statistics and write-up paragraph."""
    date_str = SCRAPE_DATE.strftime("%B %d, %Y")

    print("\n" + "=" * 65)
    print("  JOB POSTINGS SUMMARY — Corroborating Evidence Only")
    print("=" * 65)

    sub_only_cnt = stats["dash_total_incl_subsidiaries"] - stats["dash_total"]
    print(f"\n  Total deduplicated postings: {len(df)}")
    print(f"    DoorDash (delivery only): {stats['dash_total']:>4}  "
          f"(merchant_sales_us={stats['dash_merchant_sales_us_total']}, "
          f"intl={stats['dash_merchant_sales_intl_total']}, "
          f"deliveroo_mkt={stats['dash_deliveroo_market_total']}, "
          f"dasher={stats['dash_dasher_supply_total']})")
    if sub_only_cnt > 0:
        print(f"    DoorDash subsidiaries (B2B SaaS, excl. from headline ratio): {sub_only_cnt}")
    print(f"    Uber Eats:                {stats['uber_total']:>4}  "
          f"(merchant_sales_us={stats['uber_merchant_sales_us_total']})")
    print(f"    Instacart:                {stats['cart_total']:>4}")
    print(f"\n  dash_vs_uber_posting_ratio:  {_fmt(stats['dash_vs_uber_posting_ratio'], '{:.2f}x')}")
    print(f"  dash_vs_cart_posting_ratio:  {_fmt(stats['dash_vs_cart_posting_ratio'], '{:.2f}x')}")
    print(f"  DASH intl share of postings: {stats['dash_intl_share_pct']:.1f}%")
    print(f"  UBER intl share of postings: {stats['uber_intl_share_pct']:.1f}%")

    print("\n  Sources used:")
    for src, n in df["source"].value_counts().items():
        print(f"    {src:<30} {n:>4} postings")

    # ── Write-up paragraph ─────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  WRITE-UP PARAGRAPH (corroborating evidence section):")
    print("-" * 65)

    if stats["dash_total"] == 0:
        # No DASH data — be honest rather than print a nonsense narrative
        paragraph = (
            f"As of {date_str}, automated DASH job posting collection failed "
            f"(Indeed returned HTTP 403; DoorDash Greenhouse token 404'd and "
            f"their Workday CXS endpoint returned HTTP 422 — no verified free "
            f"API source available). Peer data: Uber {stats['uber_total']} "
            f"relevant postings, Instacart {stats['cart_total']} relevant "
            f"postings. The job postings signal is unavailable for DASH this "
            f"run — flag as Future Work in the write-up and corroborate the "
            f"international expansion thesis via DoorDash careers page manual "
            f"review (careers.doordash.com) plus Q4 2025 transcript guidance."
        )
    else:
        # Have DASH data — assemble the comparative narrative
        uber_clause = (
            f"vs. {stats['uber_merchant_sales_us_total']} at Uber"
            if stats["uber_total"] > 0
            else "(Uber data unavailable for direct comparison)"
        )
        cart_clause = (
            f" and {stats['cart_total']} total at Instacart"
            if stats["cart_total"] > 0 else ""
        )
        # Distinguish "no UBER data → can't compare" from "ratio is 0.0x"
        intl_ratio_val = stats.get("intl_ratio_vs_uber")
        if intl_ratio_val is None or (
            isinstance(intl_ratio_val, float) and np.isnan(intl_ratio_val)
        ):
            intl_clause = "UBER international comparison unavailable"
        else:
            intl_clause = f"{intl_ratio_val:.1f}x UBER's equivalent ratio"
        pace_clause = (
            f"{stats['dash_vs_uber_posting_ratio']:.1f}x the pace of Uber"
            if stats["uber_total"] > 0
            else "at a pace not directly comparable to Uber (Uber data unavailable)"
        )
        dasher_clause = (
            "active supply-growth investment"
            if stats["dash_dasher_supply_total"] > 0
            else "limited net-new Dasher acquisition spend"
        )

        # Honest read of the international signal — corroborates or contradicts
        # the management Q2-Q3 2026 international-acceleration thesis depending
        # on what the data actually shows.
        d_intl = stats["dash_intl_share_pct"]
        u_intl = stats["uber_intl_share_pct"]
        if d_intl >= u_intl and d_intl > 5:
            intl_thesis = (
                "supportive of management's guided acceleration in international "
                "GOV for Q2-Q3 2026"
            )
        elif d_intl > 0 and d_intl < u_intl:
            intl_thesis = (
                f"weaker than the {u_intl:.0f}% intl share at Uber Eats — directionally "
                "consistent with international acceleration but smaller than peer "
                "intensity would suggest"
            )
        else:
            intl_thesis = (
                f"a notable absence given management's guided Q2-Q3 2026 international "
                f"acceleration — Uber Eats shows {u_intl:.0f}% intl share, suggesting "
                "DASH international hiring is either routed through Deliveroo's own "
                "channels or the international ramp has not yet begun"
            )

        paragraph = (
            f"As of {date_str}, DoorDash has {stats['dash_merchant_sales_us_total']} "
            f"active US merchant sales roles {uber_clause}{cart_clause}. "
            f"International (Deliveroo market: UK/DE/FR/AU) roles account for "
            f"{d_intl:.1f}% of DASH postings — {intl_clause} — {intl_thesis}. "
            f"DASH Dasher supply/acquisition postings total "
            f"{stats['dash_dasher_supply_total']}, suggesting {dasher_clause}. "
            f"Overall, DASH is hiring at {pace_clause} across tracked role types "
            f"(Uber filtered to Delivery business line for apples-to-apples)."
        )

    # Wrap at ~78 chars for readability
    words = paragraph.split()
    line, out = [], []
    for w in words:
        line.append(w)
        if sum(len(x) + 1 for x in line) >= 78:
            out.append("  " + " ".join(line))
            line = []
    if line:
        out.append("  " + " ".join(line))
    print("\n".join(out))
    print("-" * 65)


# ── Fallback CSV ───────────────────────────────────────────────────────────────

def _write_fallback_csv() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    fallback = pd.DataFrame(
        [{"signal_available": False}],
        columns=OUTPUT_COLS,
    ).fillna(pd.NA)
    fallback["signal_available"] = False
    fallback.to_csv(JOB_POSTINGS_PATH, index=False)
    print("  Wrote empty fallback job_postings.csv.")


# ── Main ───────────────────────────────────────────────────────────────────────

def pull_job_postings() -> pd.DataFrame:
    """
    Full pipeline:
      1. Indeed HTML scraping (all companies, US + international for DASH)
      2. Greenhouse API (DoorDash careers spot check + peer fallback)
      3. Merge, deduplicate, classify
      4. Print summary statistics and write-up paragraph

    Returns deduplicated DataFrame. Corroborating evidence only —
    never pass these columns to the OLS feature selector.
    """
    print("\n" + "=" * 65)
    print("  Job Postings Pull — Corroborating Evidence Only")
    print("  NOT a model feature: forward-looking investment thesis corroboration")
    print(f"  Scrape date: {SCRAPE_DATE.date()}")
    print("=" * 65)

    # ── Step 1: Indeed scraping ────────────────────────────────────────────────
    print("\n--- Source 1: Indeed (primary) ---")
    indeed_rows = pull_indeed_all()
    print(f"\n  Indeed total raw results: {len(indeed_rows)}")

    # ── Step 2: Greenhouse API — DASH (doordashusa) + CART (instacart) ────────
    print("\n--- Source 2: Greenhouse API (DASH + CART) ---")
    gh_rows = pull_greenhouse_all()
    print(f"\n  Greenhouse total raw results: {len(gh_rows)}")

    # ── Step 3: Ashby — Deliveroo (DoorDash subsidiary) ───────────────────────
    print("\n--- Source 3: Ashby API — Deliveroo (DASH subsidiary) ---")
    print("  [Ashby | Deliveroo] Pulling careers API …", flush=True)
    ashby_rows = _pull_ashby_deliveroo()
    print(f"  Ashby/Deliveroo: {len(ashby_rows)} relevant rows (one per job-location)")

    # ── Step 4: Uber careers API ──────────────────────────────────────────────
    print("\n--- Source 4: Uber careers API (www.uber.com/api/loadSearchJobsResults) ---")
    print("  [Uber] Pulling careers API …", flush=True)
    uber_rows = _pull_uber_careers()
    print(f"  Uber: {len(uber_rows)} relevant postings (one row per job-location)")

    all_rows = indeed_rows + gh_rows + ashby_rows + uber_rows
    if not all_rows:
        return pd.DataFrame(columns=OUTPUT_COLS)

    # ── Step 3: Build DataFrame, classify, deduplicate ─────────────────────────
    df = pd.DataFrame(all_rows)

    # Ensure all output columns present
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["date_scraped"] = SCRAPE_DATE
    df["signal_available"] = True
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")

    # Cap date_posted at scrape date (no future dates from malformed parsing)
    df["date_posted"] = df["date_posted"].clip(upper=SCRAPE_DATE)

    # Validate: both Indeed and Greenhouse may duplicate the same role
    print("\n  Deduplicating across sources …")
    df = _deduplicate(df)

    # ── Step 4: Summary ────────────────────────────────────────────────────────
    stats = _compute_summary(df)
    _print_summary(stats, df)

    return df[OUTPUT_COLS]


def save_job_postings() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    try:
        df = pull_job_postings()
    except Exception as e:
        print(f"\n  Job postings pull failed: {e}")
        print("  Job postings pull failed. Using as Future Work only.")
        _write_fallback_csv()
        return

    if df.empty:
        print("\nJob postings pull failed. Using as Future Work only.")
        _write_fallback_csv()
        return

    df.to_csv(JOB_POSTINGS_PATH, index=False)
    print(f"\nSaved → {JOB_POSTINGS_PATH}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Shape:   {df.shape}")
    print(f"Date range: {df['date_posted'].min().date()} → {df['date_posted'].max().date()}")

    # Per-company missing-value check
    for company in ["DoorDash", "Uber", "Instacart"]:
        mask = df["company"] == company
        if mask.any():
            n_null = df.loc[mask, "date_posted"].isna().sum()
            print(f"  {company}: {mask.sum()} rows, {n_null} null date_posted")


if __name__ == "__main__":
    save_job_postings()
