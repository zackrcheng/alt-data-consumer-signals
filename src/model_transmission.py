"""
model_transmission.py — DASH transmission chain (GOV → revenue → EBITDA → CAR).

Three regressions form the chain:
  β1  rev_surprise_pct        ~ gov_surprise_pct                 (DASH-only OLS)
  β2  ebitda_margin_chg_yoy   ~ rev_surprise_pct                 (DASH-only OLS)
       + DASH+CART panel      ~ revenue_yoy_pct  (robustness; CART has no IBES)
  β3  CAR[-1,+2]              ~ gov_surprise_pct                 (Session 13)

Chain applied to the published Q1 2026 GOV surprise prediction (read from
outputs/tables/q1_2026_preregistered.csv) propagates point + 80% CI through
β1·β2 to implied revenue surprise and EBITDA margin lift.

Also runs a variance decomposition of the surprise target into feature
groups (Trends, AppStore, macro, autoregressive) via sequential-R².

Outputs:
  outputs/tables/transmission_betas.csv         β1, β2 with SE / 95%CI
  outputs/tables/transmission_chain_q1_2026.csv  chain applied to forecast
  outputs/tables/variance_decomposition.csv      seq-R² per feature group
  outputs/figures/transmission_chain.png         scatter + fit per regression
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    MASTER_DF_PATH, COMPUSTAT_PATH,
    OUTPUTS_TABLES, OUTPUTS_FIGURES,
    CHART_STYLE, COLORS, RANDOM_SEED,
)


PREREG_PATH = OUTPUTS_TABLES / "q1_2026_preregistered.csv"


# ── Data prep ────────────────────────────────────────────────────────────────

def prepare_dash_panel(master_df: pd.DataFrame) -> pd.DataFrame:
    """DASH-only panel with the full chain variables."""
    df = master_df.sort_values("quarter_end_date").reset_index(drop=True).copy()
    df["revenue_yoy_pct"] = df["revenue_actual_bn"].pct_change(4, fill_method=None) * 100
    df["ebitda_margin_chg_yoy_pp"] = df["ebitda_margin_pct"] - df["ebitda_margin_pct"].shift(4)
    df["ebitda_margin_chg_qoq_pp"] = df["ebitda_margin_pct"].diff()
    keep = [
        "quarter_label", "quarter_end_date",
        "gov_surprise_pct", "rev_surprise_pct",
        "revenue_actual_bn", "revenue_yoy_pct",
        "ebitda_actual_bn", "ebitda_margin_pct",
        "ebitda_margin_chg_yoy_pp", "ebitda_margin_chg_qoq_pp",
    ]
    return df[keep].copy()


def prepare_cart_panel(compustat_path: Path = COMPUSTAT_PATH) -> pd.DataFrame:
    """CART panel for the operating-leverage robustness check (Q4 2023+).

    CART has no IBES coverage in this project, so we use realized YoY revenue
    growth instead of consensus-surprise. Q3 2023 EBITDA margin was -320%
    (pre-IPO weirdness) — we drop pre-Q4 2023 rows.
    """
    df = pd.read_csv(compustat_path, parse_dates=["quarter_end_date"])
    cart = df[df["ticker"] == "CART"].sort_values("quarter_end_date").copy()
    cart["revenue_yoy_pct"] = cart["revenue_bn"].pct_change(4, fill_method=None) * 100
    cart["ebitda_margin_chg_yoy_pp"] = (
        cart["ebitda_margin_pct"] - cart["ebitda_margin_pct"].shift(4)
    )
    cart["ticker"] = "CART"
    cart = cart[cart["quarter_label"] >= "Q4_2023"].copy()
    return cart[["ticker", "quarter_label", "revenue_yoy_pct",
                 "ebitda_margin_chg_yoy_pp"]].copy()


# ── Regression helpers ───────────────────────────────────────────────────────

def _ols_summary(fit, x_col: str) -> dict:
    """Pack a statsmodels OLS fit into a flat dict."""
    coef = float(fit.params[x_col])
    se = float(fit.bse[x_col])
    p = float(fit.pvalues[x_col])
    ci_low, ci_high = fit.conf_int(alpha=0.05).loc[x_col].values
    return {
        "beta": coef, "stderr": se, "p_value": p,
        "ci95_lo": float(ci_low), "ci95_hi": float(ci_high),
        "intercept": float(fit.params["const"]),
        "r_squared": float(fit.rsquared),
        "n": int(fit.nobs),
    }


def regression_b1(dash: pd.DataFrame) -> dict:
    """β1: rev_surprise_pct ~ gov_surprise_pct (DASH only). Pass-through."""
    sub = dash.dropna(subset=["gov_surprise_pct", "rev_surprise_pct"])
    X = sm.add_constant(sub[["gov_surprise_pct"]])
    y = sub["rev_surprise_pct"]
    fit = sm.OLS(y, X).fit()
    return {"name": "b1_gov_to_rev_surprise",
            **_ols_summary(fit, "gov_surprise_pct"), "fit": fit}


def regression_b1_robust(dash: pd.DataFrame, exclude_quarters=("Q4_2025",)) -> dict:
    """β1 with the Deliveroo-consolidation quarter excluded. Q4 2025 was the
    first DASH quarter consolidating Deliveroo for ~half the period; revenue
    consensus hadn't fully priced it in, so the rev surprise (+11pp) is much
    larger than the underlying take-rate-driven response."""
    sub = dash[~dash["quarter_label"].isin(exclude_quarters)].dropna(
        subset=["gov_surprise_pct", "rev_surprise_pct"])
    X = sm.add_constant(sub[["gov_surprise_pct"]])
    y = sub["rev_surprise_pct"]
    fit = sm.OLS(y, X).fit()
    out = {"name": "b1_gov_to_rev_surprise_ex_Q4_2025",
           **_ols_summary(fit, "gov_surprise_pct"), "fit": fit,
           "excluded": list(exclude_quarters)}
    return out


def regression_b2(dash: pd.DataFrame) -> dict:
    """β2: ebitda_margin_chg_yoy_pp ~ rev_surprise_pct (DASH only)."""
    sub = dash.dropna(subset=["rev_surprise_pct", "ebitda_margin_chg_yoy_pp"])
    X = sm.add_constant(sub[["rev_surprise_pct"]])
    y = sub["ebitda_margin_chg_yoy_pp"]
    fit = sm.OLS(y, X).fit()
    return {"name": "b2_rev_surprise_to_ebitda_margin_yoy_pp",
            **_ols_summary(fit, "rev_surprise_pct"), "fit": fit}


def regression_b2_panel(dash: pd.DataFrame, cart: pd.DataFrame) -> dict:
    """β2 on a DASH+CART panel using realized YoY revenue growth (CART has
    no IBES surprise). Adds a CART fixed effect."""
    dash_p = dash.copy()
    dash_p["ticker"] = "DASH"
    dash_p = dash_p[["ticker", "quarter_label", "revenue_yoy_pct",
                     "ebitda_margin_chg_yoy_pp"]]
    panel = pd.concat([dash_p, cart], ignore_index=True).dropna(
        subset=["revenue_yoy_pct", "ebitda_margin_chg_yoy_pp"])
    panel["is_cart"] = (panel["ticker"] == "CART").astype(float)
    X = sm.add_constant(panel[["revenue_yoy_pct", "is_cart"]])
    y = panel["ebitda_margin_chg_yoy_pp"]
    fit = sm.OLS(y, X).fit()
    return {"name": "b2_rev_yoy_to_ebitda_margin_yoy_pp_DASH_CART_panel",
            **_ols_summary(fit, "revenue_yoy_pct"),
            "is_cart_coef": float(fit.params["is_cart"]),
            "fit": fit, "panel_n_dash": int((panel["ticker"]=="DASH").sum()),
            "panel_n_cart": int((panel["ticker"]=="CART").sum())}


# ── Variance decomposition ──────────────────────────────────────────────────

def variance_decomposition(master_df: pd.DataFrame, target: str = "gov_surprise_pct"
                            ) -> pd.DataFrame:
    """Sequential-R²: starting from intercept-only, add feature groups in
    a fixed order and report the marginal R² lift each group contributes."""
    # Group order: alt-data demand-side first, then macro, then autoregressive
    GROUPS = [
        ("Trends",   ["doordash_trends_momentum", "four_way_doordash_share_mean"]),
        ("AppStore", ["dash_engagement_x_sentiment_mean", "dash_net_sentiment_mean"]),
        ("Macro",    ["consumer_health_index", "jolts_transport_yoy"]),
        ("Autoregressive", ["prior_qtr_gov_surprise_pct"]),
    ]
    all_feats = [f for _, fs in GROUPS for f in fs if f in master_df.columns]
    sub = master_df.dropna(subset=[target] + all_feats).copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    cumulative_features: list[str] = []
    last_r2 = 0.0
    for group_label, feats in GROUPS:
        feats_present = [f for f in feats if f in sub.columns]
        cumulative_features += feats_present
        if not cumulative_features:
            continue
        X = sm.add_constant(sub[cumulative_features])
        fit = sm.OLS(sub[target], X).fit()
        r2 = float(fit.rsquared)
        rows.append({
            "group":             group_label,
            "features_added":    ", ".join(feats_present),
            "n_features_total":  len(cumulative_features),
            "cumulative_r2":     round(r2, 3),
            "marginal_r2":       round(r2 - last_r2, 3),
        })
        last_r2 = r2
    out = pd.DataFrame(rows)
    out["pct_variance_added"] = (out["marginal_r2"] * 100).round(1)
    return out


# ── Chain application ───────────────────────────────────────────────────────

def apply_chain(b1: dict, b2: dict, prereg: pd.Series) -> dict:
    """Propagate the published Q1 2026 GOV surprise + 80% CI through β1·β2.
    When β1 or β2 is negative, the CI bounds flip direction — we sort
    [ci80_lo, ci80_hi] so lo ≤ hi for display."""
    gov_pt = float(prereg["q1_2026_pred_pct"])
    gov_lo = float(prereg["q1_2026_ci_80_lo"])
    gov_hi = float(prereg["q1_2026_ci_80_hi"])

    # β1 application: rev_surprise = α1 + β1 × gov_surprise
    rev_pt = b1["intercept"] + b1["beta"] * gov_pt
    rev_a  = b1["intercept"] + b1["beta"] * gov_lo
    rev_b  = b1["intercept"] + b1["beta"] * gov_hi
    rev_lo, rev_hi = (rev_a, rev_b) if rev_a <= rev_b else (rev_b, rev_a)

    # β2 application: ebitda_margin_chg = α2 + β2 × rev_surprise
    margin_pt = b2["intercept"] + b2["beta"] * rev_pt
    margin_a  = b2["intercept"] + b2["beta"] * rev_lo
    margin_b  = b2["intercept"] + b2["beta"] * rev_hi
    margin_lo, margin_hi = (margin_a, margin_b) if margin_a <= margin_b else (margin_b, margin_a)

    return {
        "gov_surprise_pp":    {"point": gov_pt, "ci80_lo": gov_lo, "ci80_hi": gov_hi},
        "rev_surprise_pp":    {"point": rev_pt, "ci80_lo": rev_lo, "ci80_hi": rev_hi},
        "ebitda_margin_chg_yoy_pp": {"point": margin_pt, "ci80_lo": margin_lo,
                                       "ci80_hi": margin_hi},
    }


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_chain(dash: pd.DataFrame, b1: dict, b2: dict, chain: dict,
                out_path: Path) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(CHART_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # β1 scatter
    ax = axes[0]
    sub = dash.dropna(subset=["gov_surprise_pct", "rev_surprise_pct"])
    ax.scatter(sub["gov_surprise_pct"], sub["rev_surprise_pct"],
               color=COLORS["dash_primary"], s=60, alpha=0.85)
    xs = np.linspace(sub["gov_surprise_pct"].min() - 1,
                      max(sub["gov_surprise_pct"].max(),
                          chain["gov_surprise_pp"]["ci80_hi"]) + 1, 50)
    ax.plot(xs, b1["intercept"] + b1["beta"] * xs,
            color=COLORS["actual"], lw=2,
            label=f"β1 = {b1['beta']:+.3f}  (R²={b1['r_squared']:.2f}, n={b1['n']})")
    # Mark Q1 2026 chain point
    g = chain["gov_surprise_pp"]; r = chain["rev_surprise_pp"]
    ax.errorbar([g["point"]], [r["point"]],
                xerr=[[g["point"]-g["ci80_lo"]], [g["ci80_hi"]-g["point"]]],
                yerr=[[max(0, r["point"]-r["ci80_lo"])],
                       [max(0, r["ci80_hi"]-r["point"])]],
                fmt="*", ms=14, color=COLORS["forecast"],
                label=f"Q1 2026 chain: gov={g['point']:+.2f}pp → rev={r['point']:+.2f}pp")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("GOV surprise (pp)")
    ax.set_ylabel("Revenue surprise (pp)")
    ax.set_title("β1: GOV surprise → revenue surprise (DASH)")
    ax.legend(fontsize=8, loc="upper left")

    # β2 scatter
    ax = axes[1]
    sub2 = dash.dropna(subset=["rev_surprise_pct", "ebitda_margin_chg_yoy_pp"])
    ax.scatter(sub2["rev_surprise_pct"], sub2["ebitda_margin_chg_yoy_pp"],
               color=COLORS["dash_primary"], s=60, alpha=0.85)
    xs = np.linspace(sub2["rev_surprise_pct"].min() - 1,
                      max(sub2["rev_surprise_pct"].max(),
                          chain["rev_surprise_pp"]["ci80_hi"]) + 1, 50)
    ax.plot(xs, b2["intercept"] + b2["beta"] * xs,
            color=COLORS["actual"], lw=2,
            label=f"β2 = {b2['beta']:+.3f}  (R²={b2['r_squared']:.2f}, n={b2['n']})")
    m = chain["ebitda_margin_chg_yoy_pp"]
    ax.errorbar([r["point"]], [m["point"]],
                xerr=[[max(0, r["point"]-r["ci80_lo"])],
                       [max(0, r["ci80_hi"]-r["point"])]],
                yerr=[[max(0, m["point"]-m["ci80_lo"])],
                       [max(0, m["ci80_hi"]-m["point"])]],
                fmt="*", ms=14, color=COLORS["forecast"],
                label=f"Q1 2026 chain: rev={r['point']:+.2f}pp → margin={m['point']:+.2f}pp")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Revenue surprise (pp)")
    ax.set_ylabel("EBITDA margin YoY change (pp)")
    ax.set_title("β2: revenue surprise → EBITDA margin lift (DASH)")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_DF_PATH, parse_dates=["quarter_end_date"])
    dash = prepare_dash_panel(master)
    cart = prepare_cart_panel()
    prereg = pd.read_csv(PREREG_PATH).iloc[0]

    # Regressions
    b1 = regression_b1(dash)
    b1_robust = regression_b1_robust(dash, exclude_quarters=("Q4_2025",))
    b2 = regression_b2(dash)
    b2_panel = regression_b2_panel(dash, cart)

    # Chain application — full sample and Deliveroo-excluded sensitivity
    chain = apply_chain(b1, b2, prereg)
    chain_robust = apply_chain(b1_robust, b2, prereg)

    # Variance decomposition (against the actual model target — surprise)
    vdec = variance_decomposition(master, target="gov_surprise_pct")

    # ── Save tables ──────────────────────────────────────────────────────────
    betas_rows = []
    for label, r in [("β1 (DASH, full)", b1),
                      ("β1 (DASH, ex Q4_2025)", b1_robust),
                      ("β2 (DASH)", b2),
                      ("β2 (DASH+CART panel)", b2_panel)]:
        betas_rows.append({
            "regression": label, "name": r["name"],
            "beta": round(r["beta"], 4), "stderr": round(r["stderr"], 4),
            "p_value": round(r["p_value"], 4),
            "ci95_lo": round(r["ci95_lo"], 4), "ci95_hi": round(r["ci95_hi"], 4),
            "intercept": round(r["intercept"], 4),
            "r_squared": round(r["r_squared"], 4),
            "n": r["n"],
        })
    betas_df = pd.DataFrame(betas_rows)
    betas_df.to_csv(OUTPUTS_TABLES / "transmission_betas.csv", index=False)

    chain_rows = []
    for stage, vals in chain.items():
        chain_rows.append({"variant": "full_sample", "stage": stage, **vals})
    for stage, vals in chain_robust.items():
        chain_rows.append({"variant": "ex_Q4_2025", "stage": stage, **vals})
    chain_df = pd.DataFrame(chain_rows)
    chain_df.to_csv(OUTPUTS_TABLES / "transmission_chain_q1_2026.csv", index=False)

    if not vdec.empty:
        vdec.to_csv(OUTPUTS_TABLES / "variance_decomposition.csv", index=False)

    # Plot
    plot_chain(dash, b1, b2, chain, OUTPUTS_FIGURES / "transmission_chain.png")

    # ── Stdout summary ───────────────────────────────────────────────────────
    print("=" * 72)
    print("TRANSMISSION CHAIN BETAS")
    print("=" * 72)
    print(betas_df.to_string(index=False))

    print()
    print("=" * 72)
    print("VARIANCE DECOMPOSITION (target: gov_surprise_pct)")
    print("=" * 72)
    if not vdec.empty:
        print(vdec.to_string(index=False))

    print()
    print("=" * 72)
    print("CHAIN APPLIED TO PRE-REGISTERED Q1 2026 GOV SURPRISE")
    print("=" * 72)
    for label, ch, b1_ in [("Full sample (incl. Q4 2025 Deliveroo step-up)",
                              chain, b1),
                             ("Sensitivity: Q4 2025 excluded",
                              chain_robust, b1_robust)]:
        print(f"\n  {label}  (β1={b1_['beta']:.3f})")
        for stage, vals in ch.items():
            print(f"    {stage:30s}  {vals['point']:+6.2f}pp  "
                  f"(80% CI [{vals['ci80_lo']:+.2f}, {vals['ci80_hi']:+.2f}])")

    print()
    print("Disclosures for the L/S note:")
    print(f"  • β1 (full) = {b1['beta']:.3f} is empirically much higher than the")
    print(f"    project-doc prior of ~0.9 (stable-take-rate pass-through). DASH revenue")
    print(f"    surprises are systematically larger than GOV surprises in this sample,")
    print(f"    driven by (a) take-rate expansion (~13.3 → 13.8% over the window),")
    print(f"    (b) ads/non-GOV revenue, and (c) a Q4 2025 Deliveroo consolidation")
    print(f"    that priced into actuals before consensus. β1 ex-Q4 2025 = "
          f"{b1_robust['beta']:.3f}.")
    print(f"  • β2 = {b2['beta']:.3f} (DASH-only) is weakly negative with R²={b2['r_squared']:.2f},")
    print(f"    p={b2['p_value']:.2f}. Operating leverage is NOT a clean story in this")
    print(f"    sample — revenue beats don't reliably translate into margin expansion")
    print(f"    YoY at this n. The DASH+CART panel β2 = {b2_panel['beta']:.3f} (also weak).")
    print(f"  • β3 (CAR sensitivity) is the Session 13 event-study output.")


if __name__ == "__main__":
    main()
