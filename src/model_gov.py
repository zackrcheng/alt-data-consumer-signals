"""
model_gov.py — DASH GOV-surprise model comparison framework.

  16 model variants  =  2 feature sets × 2 targets × 4 architectures
  + 2 baselines (zero, trailing-4q mean)

The comparison is selected on directional_acc (tiebreak rmse), then a
quantile regression is fit on the chosen variant for the Q1 2026 80% CI.
The result is written to a pre-registration CSV before earnings.

Inputs (no look-ahead):
  data/processed/master_df.csv         spine, features, targets
  data/processed/dash_gov_master.csv   contribution_margin history
  data/processed/uber_gov_master.csv   UBER delivery + total GB history
  data/raw/google_trends.csv           weekly DoorDash index (for slope)

Targets:
  TARGET_RAW = gov_surprise_pct
  TARGET_STD = gov_surprise_pct / expanding_std,
                where expanding_std = surprise.expanding(min_periods=4).std().shift(1)
                (causal: at row t, divisor uses surprise[0..t-1] only)

Feature sets:
  CURRENT  = MODEL_FEATURE_COLS  (the 6 candidates from EDA Session 9)
  EXTENDED = CURRENT + [
      guidance_width_pct                 (mgmt's own uncertainty)
      uber_delivery_surprise_lag1        (peer signal — known before quarter)
      trends_slope                       (within-window trajectory)
      contribution_margin_delta_lag1     (margin direction last quarter)
  ]
  All extended features are computed in this script with explicit lags
  so the value at row t uses only data available before quarter_end_date[t].

Q1 2026 forecast: jolts_transport_yoy is missing (FRED publication lag).
We impute it with the trailing-4q mean from training data only — applied
to the Q1 2026 row before any model fit (Session 9 decision).

Outputs:
  outputs/tables/model_comparison.csv       all 16 variants + baselines
  outputs/tables/q1_2026_preregistered.csv  selected variant + CI
  outputs/figures/walk_forward.png          actual vs predicted (top variant)
"""

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import (
    MASTER_DF_PATH, DASH_GOV_MASTER_PATH, UBER_GOV_MASTER_PATH,
    GOOGLE_TRENDS_PATH, FORECAST_QUARTER, MODEL_FEATURE_COLS, CORROBORATING_COLS,
    WALK_FORWARD_MIN_TRAIN_QUARTERS, OUTPUTS_TABLES, OUTPUTS_FIGURES,
    RANDOM_SEED, TRENDS_WINDOW_WEEKS, TRENDS_LAG_WEEKS, QUARTER_END_DATES,
    CHART_STYLE, COLORS,
)

np.random.seed(RANDOM_SEED)


# ── Constants ────────────────────────────────────────────────────────────────

CURRENT_FEATURES = list(MODEL_FEATURE_COLS)            # 6 features
EXTENDED_FEATURE_ADDS = [
    "guidance_width_pct",
    "uber_delivery_surprise_lag1",
    "trends_slope",
    "contribution_margin_delta_lag1",
]
EXTENDED_FEATURES = CURRENT_FEATURES + EXTENDED_FEATURE_ADDS

TARGET_RAW = "gov_surprise_pct"
TARGET_STD = "gov_surprise_std"
# Demeaned targets force features to predict the residual on top of a
# causal base-rate estimate. Both shift(1) the divisor / subtractor so
# only past data is used.
TARGET_DEMEAN_4Q        = "gov_surprise_demean_4q"
TARGET_DEMEAN_EXPANDING = "gov_surprise_demean_expanding"
TARGETS_ALL = [TARGET_RAW, TARGET_STD, TARGET_DEMEAN_4Q, TARGET_DEMEAN_EXPANDING]

MIN_TRAIN_QUARTERS = WALK_FORWARD_MIN_TRAIN_QUARTERS   # 8
MIN_VALID_TRAIN_ROWS = 6
N_BOOTSTRAP = 500
VIF_THRESHOLD = 10.0


# ── Disjointness invariant ───────────────────────────────────────────────────

assert set(CURRENT_FEATURES).issubset(set(MODEL_FEATURE_COLS)), \
    "CURRENT_FEATURES must be a subset of MODEL_FEATURE_COLS"
assert not (set(EXTENDED_FEATURES) & set(CORROBORATING_COLS)), \
    f"EXTENDED features collide with CORROBORATING_COLS: " \
    f"{set(EXTENDED_FEATURES) & set(CORROBORATING_COLS)}"


# ── Feature engineering ──────────────────────────────────────────────────────

def compute_extended_features(master: pd.DataFrame, dash_gov: pd.DataFrame,
                               uber_gov: pd.DataFrame, trends: pd.DataFrame
                               ) -> tuple[pd.DataFrame, list[str]]:
    """Add the 4 EXTENDED features. All lags chosen so feature value at
    row t uses only data available before quarter_end_date[t]."""
    df = master.copy()
    added = []

    # 1. guidance_width_pct
    width = (df["gov_guidance_high_mn"] - df["gov_guidance_low_mn"])
    df["guidance_width_pct"] = (width / df["gov_guidance_mid_mn"]) * 100.0
    added.append("guidance_width_pct")

    # 2. uber_delivery_surprise_lag1
    # Reconstruct delivery-segment-implied consensus from total-GB consensus
    # and the prior-quarter delivery mix (mix is *known* before quarter t).
    u = uber_gov.sort_values("quarter_end_date").set_index("quarter_label").copy()
    delivery_mix = u["gb_delivery_actual_mn"] / u["gb_total_actual_mn"]
    delivery_mix_lag1 = delivery_mix.shift(1)
    implied_consensus = u["gb_total_factset_consensus_mn"] * delivery_mix_lag1
    surprise = (u["gb_delivery_actual_mn"] - implied_consensus) / implied_consensus * 100.0
    u["uber_delivery_surprise_lag1"] = surprise.shift(1)
    df = df.merge(
        u[["uber_delivery_surprise_lag1"]].reset_index(),
        on="quarter_label", how="left",
    )
    added.append("uber_delivery_surprise_lag1")

    # 3. trends_slope — slope of weekly DoorDash index over the 8-week
    #    pre-quarter window (same window used by quarterly mean features).
    t = trends.copy()
    t["date"] = pd.to_datetime(t["date"])
    slopes = {}
    for q_label, qe_str in QUARTER_END_DATES.items():
        qe = pd.Timestamp(qe_str)
        win_end = qe - pd.Timedelta(weeks=TRENDS_LAG_WEEKS)
        win_start = win_end - pd.Timedelta(weeks=TRENDS_WINDOW_WEEKS)
        sub = t.loc[(t["date"] >= win_start) & (t["date"] < win_end),
                    ["date", "DoorDash"]].dropna()
        if len(sub) < 4:
            slopes[q_label] = np.nan
            continue
        weeks = ((sub["date"] - sub["date"].min()).dt.days / 7.0).values
        idx = sub["DoorDash"].values.astype(float)
        slopes[q_label] = float(np.polyfit(weeks, idx, 1)[0])
    df["trends_slope"] = df["quarter_label"].map(slopes)
    added.append("trends_slope")

    # 4. contribution_margin_delta_lag1 — at row t, this is
    #    (margin[t-1] - margin[t-2]). Built from the dash_gov_master series
    #    and shifted so it's strictly observed-before-t.
    g = dash_gov.sort_values("quarter_end_date").set_index("quarter_label").copy()
    cm_qoq = g["contribution_margin_pct"].diff()      # current - prior
    cm_qoq_lag1 = cm_qoq.shift(1)                     # row t = QoQ as of t-1
    df = df.merge(
        cm_qoq_lag1.rename("contribution_margin_delta_lag1").reset_index(),
        on="quarter_label", how="left",
    )
    added.append("contribution_margin_delta_lag1")

    return df, added


def add_alternative_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three causal alternative targets:
       • TARGET_STD              — surprise / causal expanding std.
       • TARGET_DEMEAN_4Q        — surprise − causal trailing-4q mean.
       • TARGET_DEMEAN_EXPANDING — surprise − causal expanding mean.
    All three .shift(1) the divisor/subtractor so only past data feeds in.
    The conversion factors (std, 4q mean, expanding mean) are stored as
    columns prefixed with `_` for use at forecast time."""
    df = df.copy()
    s = df[TARGET_RAW]

    expanding_std = s.expanding(min_periods=4).std().shift(1)
    df[TARGET_STD] = s / expanding_std
    df["_expanding_std_for_surprise"] = expanding_std

    trail_4q_mean = s.rolling(4).mean().shift(1)
    df[TARGET_DEMEAN_4Q] = s - trail_4q_mean
    df["_demean_4q_baseline"] = trail_4q_mean

    expanding_mean = s.expanding(min_periods=4).mean().shift(1)
    df[TARGET_DEMEAN_EXPANDING] = s - expanding_mean
    df["_demean_expanding_baseline"] = expanding_mean

    return df


# Back-compat alias — older notebook cells imported this name
add_standardized_target = add_alternative_targets


def convert_target_pred_to_pp(target: str, raw_pred: float, df: pd.DataFrame,
                                fc_idx: int) -> float:
    """Convert a prediction in target units to surprise pp at the forecast row."""
    if pd.isna(raw_pred):
        return np.nan
    if target == TARGET_RAW:
        return float(raw_pred)
    if target == TARGET_STD:
        # forecast-time expanding std: std of all historical surprise through
        # the row strictly before the forecast row.
        hist = df.iloc[:fc_idx][TARGET_RAW].dropna()
        std = float(hist.expanding(min_periods=4).std().iloc[-1])
        return float(raw_pred) * std
    if target == TARGET_DEMEAN_4Q:
        baseline = float(df["_demean_4q_baseline"].iloc[fc_idx])
        return float(raw_pred) + baseline
    if target == TARGET_DEMEAN_EXPANDING:
        baseline = float(df["_demean_expanding_baseline"].iloc[fc_idx])
        return float(raw_pred) + baseline
    raise ValueError(f"Unknown target: {target}")


def impute_forecast_row_jolts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Session-9 imputation rule: any feature NaN in the Q1 2026
    forecast row gets the trailing-4q mean from the historical series.
    Only the FORECAST row is touched — training data is unchanged."""
    df = df.copy()
    fc_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]
    hist = df.iloc[:fc_idx]
    for c in EXTENDED_FEATURES:
        if c not in df.columns:
            continue
        if pd.isna(df.at[fc_idx, c]):
            tail = hist[c].dropna().tail(4)
            if len(tail):
                df.at[fc_idx, c] = float(tail.mean())
    return df


# ── Models ───────────────────────────────────────────────────────────────────

class OLSDrop:
    """OLS with iterative VIF dropping (highest VIF removed until all < threshold)."""
    def __init__(self, vif_threshold: float = VIF_THRESHOLD):
        self.vif_threshold = vif_threshold
        self.dropped_features: list[str] = []
        self.kept_features: list[str] = []
        self.fit_result = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "OLSDrop":
        X = X_train.copy()
        dropped = []
        while X.shape[1] >= 2:
            X_const = sm.add_constant(X, has_constant="add")
            vifs = []
            for col in X.columns:
                idx = X_const.columns.get_loc(col)
                vifs.append((col, variance_inflation_factor(X_const.values, idx)))
            max_col, max_vif = max(vifs, key=lambda kv: kv[1])
            if not np.isfinite(max_vif) or max_vif >= self.vif_threshold:
                X = X.drop(columns=[max_col])
                dropped.append(max_col)
            else:
                break
        self.dropped_features = dropped
        self.kept_features = list(X.columns)
        self.fit_result = sm.OLS(y_train.values, sm.add_constant(X, has_constant="add")).fit()
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X = X_test[self.kept_features]
        return self.fit_result.predict(sm.add_constant(X, has_constant="add")).values


class PCAModel:
    """StandardScaler → PCA(n=min(n_features, n_samples//3)) → OLS on PCs."""
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.fit_result = None
        self.feature_cols: list[str] = []
        self.explained_variance_ratio_ = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PCAModel":
        n_samples, n_features = X_train.shape
        nc = max(1, min(n_features, n_samples // 3))
        self.scaler = StandardScaler()
        X_z = self.scaler.fit_transform(X_train)
        self.pca = PCA(n_components=nc, random_state=RANDOM_SEED)
        Z = self.pca.fit_transform(X_z)
        Z_df = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(nc)])
        self.fit_result = sm.OLS(y_train.values, sm.add_constant(Z_df)).fit()
        self.feature_cols = list(X_train.columns)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X = X_test[self.feature_cols]
        Z = self.pca.transform(self.scaler.transform(X))
        Z_df = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])])
        return self.fit_result.predict(sm.add_constant(Z_df, has_constant="add")).values


class PLSModel:
    """StandardScaler → PLSRegression(n_components=2). Maximizes covariance with target."""
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler = None
        self.pls = None
        self.feature_cols: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PLSModel":
        n_samples, n_features = X_train.shape
        nc = max(1, min(self.n_components, n_features, n_samples - 1))
        self.scaler = StandardScaler()
        X_z = self.scaler.fit_transform(X_train)
        self.pls = PLSRegression(n_components=nc)
        self.pls.fit(X_z, y_train.values.reshape(-1, 1))
        self.feature_cols = list(X_train.columns)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X = X_test[self.feature_cols]
        X_z = self.scaler.transform(X)
        return self.pls.predict(X_z).flatten()


class RidgeModel:
    """StandardScaler → RidgeCV. Records selected alpha."""
    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        self.alphas = list(alphas)
        self.scaler = None
        self.ridge = None
        self.alpha_ = None
        self.feature_cols: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeModel":
        n = len(y_train)
        cv = max(2, min(5, n))
        self.scaler = StandardScaler()
        X_z = self.scaler.fit_transform(X_train)
        self.ridge = RidgeCV(alphas=self.alphas, cv=cv)
        self.ridge.fit(X_z, y_train.values)
        self.alpha_ = float(self.ridge.alpha_)
        self.feature_cols = list(X_train.columns)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X = X_test[self.feature_cols]
        X_z = self.scaler.transform(X)
        return self.ridge.predict(X_z)


MODEL_CLASSES: dict[str, type] = {
    "ols_drop": OLSDrop,
    "pca": PCAModel,
    "pls": PLSModel,
    "ridge": RidgeModel,
}


# ── Walk-forward driver ──────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, features: list[str], target: str,
                     model_class: type) -> pd.DataFrame:
    """Drop NaN rows from training; skip a quarter if test row is NaN
    or if too few valid training rows remain."""
    rows = []
    for t in range(MIN_TRAIN_QUARTERS, len(df)):
        actual = df[target].iloc[t]
        if pd.isna(actual):
            continue
        train_X = df[features].iloc[:t]
        train_y = df[target].iloc[:t]
        valid = train_X.notna().all(axis=1) & train_y.notna()
        Xtr, ytr = train_X[valid], train_y[valid]
        if len(Xtr) < MIN_VALID_TRAIN_ROWS:
            continue
        test_X = df[features].iloc[[t]]
        if test_X.isna().any().any():
            continue
        try:
            model = model_class().fit(Xtr, ytr)
            pred = float(model.predict(test_X)[0])
        except Exception as e:
            continue
        rows.append({
            "quarter_label": df["quarter_label"].iloc[t],
            "predicted": pred,
            "actual": float(actual),
        })
    return pd.DataFrame(rows)


def evaluate(wf: pd.DataFrame) -> dict:
    if wf.empty:
        return {"rmse": np.nan, "mae": np.nan, "directional_acc": np.nan, "n_valid": 0}
    err = wf["predicted"] - wf["actual"]
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae":  float(err.abs().mean()),
        "directional_acc": float((np.sign(wf["predicted"]) == np.sign(wf["actual"])).mean()),
        "n_valid": len(wf),
    }


# ── Q1 2026 prediction with bootstrap CI ─────────────────────────────────────

def predict_q1_2026(df: pd.DataFrame, features: list[str], target: str,
                     model_class: type, n_boot: int = N_BOOTSTRAP) -> dict:
    fc_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]
    train = df.iloc[:fc_idx]
    test = df.iloc[[fc_idx]]
    valid = train[features].notna().all(axis=1) & train[target].notna()
    Xtr, ytr = train[features][valid], train[target][valid]
    if len(Xtr) < MIN_VALID_TRAIN_ROWS or test[features].isna().any().any():
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    try:
        model = model_class().fit(Xtr, ytr)
        point = float(model.predict(test[features])[0])
    except Exception:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(Xtr)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = Xtr.iloc[idx].reset_index(drop=True)
        yb = ytr.iloc[idx].reset_index(drop=True)
        try:
            mb = model_class().fit(Xb, yb)
            boots.append(float(mb.predict(test[features])[0]))
        except Exception:
            continue
    if not boots:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan}
    return {
        "point": point,
        "ci_lo": float(np.percentile(boots, 10)),
        "ci_hi": float(np.percentile(boots, 90)),
    }


# ── Baselines ────────────────────────────────────────────────────────────────

def baseline_zero_walk_forward(df: pd.DataFrame, target: str = TARGET_RAW) -> pd.DataFrame:
    rows = []
    for t in range(MIN_TRAIN_QUARTERS, len(df)):
        actual = df[target].iloc[t]
        if pd.isna(actual):
            continue
        rows.append({"quarter_label": df["quarter_label"].iloc[t],
                     "predicted": 0.0, "actual": float(actual)})
    return pd.DataFrame(rows)


def baseline_trail4q_walk_forward(df: pd.DataFrame, target: str = TARGET_RAW) -> pd.DataFrame:
    rows = []
    for t in range(MIN_TRAIN_QUARTERS, len(df)):
        actual = df[target].iloc[t]
        if pd.isna(actual):
            continue
        window = df[target].iloc[max(0, t - 4):t].dropna()
        if not len(window):
            continue
        rows.append({"quarter_label": df["quarter_label"].iloc[t],
                     "predicted": float(window.mean()),
                     "actual": float(actual)})
    return pd.DataFrame(rows)


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_top_variant(df: pd.DataFrame, top_features: list[str], top_target: str,
                     top_model_cls: type, top_name: str, q1_pred: dict,
                     out_path: Path) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(CHART_STYLE)

    wf = run_walk_forward(df, top_features, top_target, top_model_cls)
    if wf.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(wf["quarter_label"], wf["actual"], "o-", lw=2.5,
            color=COLORS["actual"], label="Actual surprise (pp or σ)")
    ax.plot(wf["quarter_label"], wf["predicted"], "^--", lw=2,
            color=COLORS["dash_primary"], label=f"Predicted ({top_name})")
    ax.axhline(0, color="grey", lw=0.5)

    pt, lo, hi = q1_pred["point"], q1_pred["ci_lo"], q1_pred["ci_hi"]
    if pd.notna(pt):
        yerr = [[pt - lo], [hi - pt]] if pd.notna(lo) and pd.notna(hi) else None
        ax.errorbar(["Q1_2026"], [pt], yerr=yerr, fmt="*", ms=14,
                    color=COLORS["forecast"], label=f"Q1 2026 = {pt:+.2f}")
    ax.set_title(f"Top variant walk-forward: {top_name}")
    ax.set_ylabel("surprise (pp or σ, depending on target)")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    master = (pd.read_csv(MASTER_DF_PATH, parse_dates=["quarter_end_date"])
                .sort_values("quarter_end_date").reset_index(drop=True))
    dash_gov = pd.read_csv(DASH_GOV_MASTER_PATH, parse_dates=["quarter_end_date"])
    uber_gov = pd.read_csv(UBER_GOV_MASTER_PATH, parse_dates=["quarter_end_date"])
    trends = pd.read_csv(GOOGLE_TRENDS_PATH, parse_dates=["date"])

    # Look-ahead guard
    assert master[master["quarter_label"] == FORECAST_QUARTER]["gov_actual_mn"].isna().all(), \
        "Look-ahead contamination: Q1 2026 actual must be NaN throughout"

    # Compute extended features
    df, added = compute_extended_features(master, dash_gov, uber_gov, trends)
    df = add_standardized_target(df)
    df = impute_forecast_row_jolts(df)

    print(f"Added EXTENDED features: {added}")
    hist = df[df["quarter_label"] != FORECAST_QUARTER]
    print(f"Coverage on historical sample (n={len(hist)}):")
    for c in added:
        n = hist[c].notna().sum()
        print(f"  {c:36s} {n}/{len(hist)} ({100*n/len(hist):3.0f}%)")
    print(f"Q1 2026 row feature presence:")
    fc_row = df[df["quarter_label"] == FORECAST_QUARTER].iloc[0]
    for c in EXTENDED_FEATURES:
        v = fc_row.get(c)
        print(f"  {c:36s} {'NaN' if pd.isna(v) else f'{v:+.3f}'}")
    print()

    # Forecast-time conversion baselines per target
    fc_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]
    surprise_hist = hist[TARGET_RAW].dropna()
    forecast_expanding_std = float(surprise_hist.expanding(min_periods=4).std().iloc[-1])
    forecast_demean_4q_baseline = float(df["_demean_4q_baseline"].iloc[fc_idx])
    forecast_demean_expanding_baseline = float(df["_demean_expanding_baseline"].iloc[fc_idx])
    print(f"Forecast-time conversion baselines (for back-out to pp):")
    print(f"  std target  •  expanding_std as of Q1 2026:           {forecast_expanding_std:.3f}pp")
    print(f"  demean_4q   •  trailing-4q mean of historical surprise: {forecast_demean_4q_baseline:.3f}pp")
    print(f"  demean_exp  •  expanding mean of historical surprise:   {forecast_demean_expanding_baseline:.3f}pp")
    print()

    # Run all variants — 2 feature sets × 4 targets × 4 architectures = 32
    feature_sets = {"current": CURRENT_FEATURES, "extended": EXTENDED_FEATURES}
    targets = TARGETS_ALL

    rows = []
    for fset_name, fset in feature_sets.items():
        for target in targets:
            for model_name, model_cls in MODEL_CLASSES.items():
                variant = f"{fset_name}__{target}__{model_name}"
                try:
                    wf = run_walk_forward(df, fset, target, model_cls)
                    metrics = evaluate(wf)
                    fcst = predict_q1_2026(df, fset, target, model_cls)
                    pred_pct = convert_target_pred_to_pp(target, fcst["point"], df, fc_idx)
                    ci_lo_pct = convert_target_pred_to_pp(target, fcst["ci_lo"], df, fc_idx)
                    ci_hi_pct = convert_target_pred_to_pp(target, fcst["ci_hi"], df, fc_idx)
                    rows.append({
                        "variant_name": variant, "target": target,
                        "features": fset_name, "model": model_name,
                        **metrics,
                        "q1_2026_pred_raw": fcst["point"],
                        "q1_2026_pred_pct": pred_pct,
                        "q1_2026_ci_80_lo": ci_lo_pct,
                        "q1_2026_ci_80_hi": ci_hi_pct,
                    })
                except Exception as e:
                    print(f"  {variant} FAILED: {e}")

    # Baselines (target = TARGET_RAW; results applicable to all variants)
    zero_wf = baseline_zero_walk_forward(df)
    zero_metrics = evaluate(zero_wf)
    trail4q_wf = baseline_trail4q_walk_forward(df)
    trail4q_metrics = evaluate(trail4q_wf)
    trail4q_q1_pred = float(df[TARGET_RAW].dropna().tail(4).mean())

    rows.append({"variant_name": "baseline_zero", "target": TARGET_RAW,
                 "features": "-", "model": "baseline",
                 **zero_metrics,
                 "q1_2026_pred_raw": 0.0, "q1_2026_pred_pct": 0.0,
                 "q1_2026_ci_80_lo": np.nan, "q1_2026_ci_80_hi": np.nan})
    rows.append({"variant_name": "baseline_trail4q", "target": TARGET_RAW,
                 "features": "-", "model": "baseline",
                 **trail4q_metrics,
                 "q1_2026_pred_raw": trail4q_q1_pred,
                 "q1_2026_pred_pct": trail4q_q1_pred,
                 "q1_2026_ci_80_lo": np.nan, "q1_2026_ci_80_hi": np.nan})

    comp = pd.DataFrame(rows)
    comp["rmse_vs_zero"] = comp["rmse"] - zero_metrics["rmse"]
    comp["rmse_vs_trail4q"] = comp["rmse"] - trail4q_metrics["rmse"]
    comp = comp.sort_values(["directional_acc", "rmse"],
                            ascending=[False, True]).reset_index(drop=True)

    comp.to_csv(OUTPUTS_TABLES / "model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by directional_acc desc, rmse asc) ===")
    print(comp.round(3).to_string(index=False))
    print()

    # Directional consensus on Q1 2026 (model variants only)
    model_only = comp[comp["model"] != "baseline"]
    signs = np.sign(model_only["q1_2026_pred_pct"].dropna())
    if len(set(signs.values.tolist())) == 1:
        direction = "beat" if signs.iloc[0] > 0 else "miss"
        print(f"Directional consensus: all {len(model_only)} variants predict {direction}.")
    else:
        n_beat = int((signs > 0).sum())
        n_miss = int((signs < 0).sum())
        n_zero = int((signs == 0).sum())
        print(f"Directional split: {n_beat} beat / {n_miss} miss / {n_zero} zero (out of {len(model_only)} variants).")
    print()

    # === Top variant selection ===
    top = model_only.iloc[0]
    second = model_only.iloc[1]
    top_features = feature_sets[top["features"]]
    top_target = top["target"]
    top_model_cls = MODEL_CLASSES[top["model"]]

    print(f"Top variant: {top['variant_name']}")
    print(f"  directional_acc = {top['directional_acc']:.3f}")
    print(f"  rmse            = {top['rmse']:.3f}")
    print(f"  n_valid         = {int(top['n_valid'])}")
    print()

    # === Quantile regression on top variant for the CI we publish ===
    fc_idx = df.index[df["quarter_label"] == FORECAST_QUARTER][0]
    train = df.iloc[:fc_idx]
    test = df.iloc[[fc_idx]]
    valid = train[top_features].notna().all(axis=1) & train[top_target].notna()
    Xtr, ytr = train[top_features][valid], train[top_target][valid]

    quant_preds: dict[float, float] = {}
    for q in (0.10, 0.50, 0.90):
        try:
            qr = QuantReg(ytr.values, sm.add_constant(Xtr, has_constant="add")).fit(q=q)
            test_X = sm.add_constant(test[top_features], has_constant="add")
            quant_preds[q] = float(qr.predict(test_X).iloc[0])
        except Exception as e:
            quant_preds[q] = np.nan

    print(f"QuantReg predictions on top variant:")
    for q, p in quant_preds.items():
        if pd.isna(p):
            print(f"  q={q:.2f}  NaN")
        else:
            disp = convert_target_pred_to_pp(top_target, p, df, fc_idx)
            print(f"  q={q:.2f}  raw = {p:+.3f}  →  {disp:+.2f}pp")
    print()

    # Convert quantile preds + CI to pp using the per-target rule
    q_pp = {q: convert_target_pred_to_pp(top_target, p, df, fc_idx)
            for q, p in quant_preds.items()}

    # === Pre-registration ===
    pred_pct = float(top["q1_2026_pred_pct"])
    ci_lo = q_pp.get(0.10, top["q1_2026_ci_80_lo"])
    ci_hi = q_pp.get(0.90, top["q1_2026_ci_80_hi"])

    direction_call = "beat" if pred_pct > 0 else ("miss" if pred_pct < 0 else "tie")
    if pd.notna(ci_lo) and ci_lo > 0:
        conviction = "high"
    elif pd.notna(ci_lo) and pd.notna(ci_hi) and ci_lo < 0 < ci_hi:
        conviction = "low"
    else:
        conviction = "low"

    rationale = (
        f"Selected by directional_acc={top['directional_acc']:.3f}, rmse={top['rmse']:.3f}. "
        f"Beat runner-up {second['variant_name']} "
        f"(dir_acc={second['directional_acc']:.3f}, rmse={second['rmse']:.3f}). "
        f"Top variant beat baseline_zero by {top['rmse_vs_zero']:+.2f}pp RMSE "
        f"and baseline_trail4q by {top['rmse_vs_trail4q']:+.2f}pp."
    )

    prereg = pd.DataFrame([{
        "timestamp":            datetime.datetime.utcnow().isoformat(),
        "selected_variant":     top["variant_name"],
        "target":               top_target,
        "feature_set":          ",".join(top_features),
        "model_architecture":   top["model"],
        "q1_2026_pred_pct":     pred_pct,
        "q1_2026_ci_80_lo":     ci_lo,
        "q1_2026_ci_80_hi":     ci_hi,
        "directional_call":     direction_call,
        "conviction":           conviction,
        "walk_forward_dir_acc": top["directional_acc"],
        "selection_rationale":  rationale,
        "note":                 "Pre-registered before Q1 2026 earnings May 6 2026",
    }])
    prereg.to_csv(OUTPUTS_TABLES / "q1_2026_preregistered.csv", index=False)

    # Plot — pass CI in the SAME units as the target the model was trained on,
    # so it overlays the walk-forward predictions cleanly.
    if top_target == TARGET_RAW:
        plot_ci_lo_raw, plot_ci_hi_raw = ci_lo, ci_hi
    elif top_target == TARGET_STD:
        plot_ci_lo_raw = ci_lo / forecast_expanding_std if pd.notna(ci_lo) else np.nan
        plot_ci_hi_raw = ci_hi / forecast_expanding_std if pd.notna(ci_hi) else np.nan
    elif top_target == TARGET_DEMEAN_4Q:
        b = forecast_demean_4q_baseline
        plot_ci_lo_raw = ci_lo - b if pd.notna(ci_lo) else np.nan
        plot_ci_hi_raw = ci_hi - b if pd.notna(ci_hi) else np.nan
    else:  # TARGET_DEMEAN_EXPANDING
        b = forecast_demean_expanding_baseline
        plot_ci_lo_raw = ci_lo - b if pd.notna(ci_lo) else np.nan
        plot_ci_hi_raw = ci_hi - b if pd.notna(ci_hi) else np.nan

    plot_top_variant(df, top_features, top_target, top_model_cls,
                     top["variant_name"],
                     {"point": top["q1_2026_pred_raw"],
                      "ci_lo": plot_ci_lo_raw, "ci_hi": plot_ci_hi_raw},
                     OUTPUTS_FIGURES / "walk_forward.png")

    print("=" * 72)
    print("PRE-REGISTRATION FILE CONTENTS  (outputs/tables/q1_2026_preregistered.csv)")
    print("=" * 72)
    for col, val in prereg.iloc[0].items():
        if isinstance(val, float):
            print(f"  {col:24s} {val:+.4f}" if pd.notna(val) else f"  {col:24s} NaN")
        else:
            print(f"  {col:24s} {val}")
    print("=" * 72)
    print()
    print("PRE-REGISTERED FORECAST (plain English):")
    print(f"  Selected variant: {top['variant_name']}")
    print(f"  Q1 2026 GOV surprise prediction: {pred_pct:+.2f}pp ({direction_call})")
    if pd.notna(ci_lo) and pd.notna(ci_hi):
        print(f"  80% CI from quantile regression: [{ci_lo:+.2f}, {ci_hi:+.2f}] pp")
    print(f"  Walk-forward directional accuracy: {top['directional_acc']:.0%} on n={int(top['n_valid'])}")
    print(f"  Conviction: {conviction}")
    print(f"  Rationale: {rationale}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
