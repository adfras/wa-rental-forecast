"""
Hierarchical logistic forecast of next-month rent rise probability by SA2.

Features include nowcasted availability_rate, churn, and rent momentum. The
model includes SA2 random intercepts. Writes:
 - data_stage/price_pressure_forecast_sa2.parquet
 - appends to data_stage/price_pressure_forecast_sa2_history.parquet
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

from src.config import (
    EVAL_DIR,
    STAGE_DIR,
    RENT_GROWTH_THRESHOLD,
    RANDOM_SEED,
    FORECAST_RECENCY_HALFLIFE,
)
from src.common.pymc_helpers import sample_nuts
from src.features.dates import compute_recency_weights, to_month
from src.features.engineering import (
    add_calendar_features,
    add_churn_proxy,
    add_rent_momentum,
    add_supply_shock_features,
    add_sparse_market_features,
    add_time_since_spike,
    apply_standardization,
    add_group_demean,
    add_interaction_features,
    compute_lodgement_weights,
    standardize_columns,
)

def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _safe_logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _winsorize_series(s: pd.Series,
                      *,
                      lower_q: float = 0.01,
                      upper_q: float = 0.99,
                      lower_bound: float | None = None,
                      upper_bound: float | None = None) -> pd.Series:
    """Winsorize a series using quantiles with optional hard bounds."""
    if s.dropna().empty:
        return s
    series = s.astype(float)
    try:
        low = float(series.quantile(lower_q))
    except Exception:
        low = np.nan
    try:
        high = float(series.quantile(upper_q))
    except Exception:
        high = np.nan
    if lower_bound is not None:
        low = lower_bound if not np.isfinite(low) else max(low, lower_bound)
    if upper_bound is not None:
        high = upper_bound if not np.isfinite(high) else min(high, upper_bound)
    if not np.isfinite(low):
        low = lower_bound
    if not np.isfinite(high):
        high = upper_bound
    if low is not None and high is not None and low > high:
        low, high = high, low
    return series.clip(lower=low, upper=high)


def _clip_standardized_matrix(X: np.ndarray,
                              *,
                              quantile: float = 0.995,
                              min_limit: float = 3.0,
                              max_limit: float = 10.0) -> np.ndarray:
    if X.size == 0:
        return X
    limits = np.nanquantile(np.abs(X), quantile, axis=0)
    limits = np.where(~np.isfinite(limits) | (limits < min_limit), min_limit, limits)
    limits = np.where(limits > max_limit, max_limit, limits)
    Xc = X.copy()
    for j, lim in enumerate(limits):
        Xc[:, j] = np.clip(Xc[:, j], -lim, lim)
    return Xc


def build_dataset(*, threshold: float = RENT_GROWTH_THRESHOLD):
    """Prepare merged features and labels for the logistic model."""
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    now = pd.read_parquet(STAGE_DIR / "availability_nowcast_sa2.parquet").copy()

    # Align month types
    sa2["month"] = to_month(sa2["month"])
    now["month"] = to_month(now["month"])

    df = sa2.merge(now, on=["sa2_code", "month"], how="left")
    # House price features (monthly medians preferred; fallback to snapshots)
    try:
        from src.features.dates import to_month as _to_month

        hp_monthly_path = STAGE_DIR / "house_prices_sa2_monthly.parquet"
        hp_snapshot_path = STAGE_DIR / "house_prices_sa2_snapshot.parquet"
        if hp_monthly_path.exists():
            hp = pd.read_parquet(hp_monthly_path).copy()
        else:
            hp = pd.read_parquet(hp_snapshot_path).copy()
        rename_cols = {}
        if "allocation_weight_sum" in hp.columns:
            rename_cols["allocation_weight_sum"] = "price_allocation_weight_sum"
        if "n_suburbs" in hp.columns:
            rename_cols["n_suburbs"] = "price_suburb_count"
        if rename_cols:
            hp = hp.rename(columns=rename_cols)
        hp = hp[[c for c in hp.columns if c in {"sa2_code", "month", "median_house_price", "price_allocation_weight_sum", "price_suburb_count"}]]
        hp["sa2_code"] = hp["sa2_code"].astype(str)
        hp["month"] = _to_month(hp["month"])  # normalize

        months_all = pd.DataFrame({"month": sorted(df["month"].unique())})

        def _expand(g: pd.DataFrame) -> pd.DataFrame:
            cols = [c for c in g.columns if c in {"month", "median_house_price", "price_allocation_weight_sum", "price_suburb_count"}]
            out = months_all.merge(g[cols], on="month", how="left").sort_values("month")
            out["median_house_price"] = out["median_house_price"].ffill()
            if "price_allocation_weight_sum" in out.columns:
                out["price_allocation_weight_sum"] = out["price_allocation_weight_sum"].ffill()
            if "price_suburb_count" in out.columns:
                out["price_suburb_count"] = out["price_suburb_count"].ffill()
            out["sa2_code"] = g["sa2_code"].iloc[0]
            return out

        if not hp.empty:
            expanded = [
                _expand(group)
                for _, group in hp.groupby("sa2_code", group_keys=False)
            ]
            hp_exp = pd.concat(expanded, ignore_index=True) if expanded else pd.DataFrame()
            if not hp_exp.empty:
                df = df.merge(hp_exp, on=["sa2_code", "month"], how="left")
        df = df.sort_values(["sa2_code","month"]).copy()
        # Derived features
        df["price_level_log"] = np.log(df["median_house_price"])  # NaN-safe
        wa_med = df.groupby("month")["median_house_price"].transform("median")
        df["price_band_high"] = (df["median_house_price"] >= wa_med).astype(float)
        df["price_yield"] = (df["median_rent"] * 52.0) / df["median_house_price"]
        # Price momentum (3-month annualized approx)
        df["price_mom_3m"] = (df["median_house_price"] / df.groupby("sa2_code")["median_house_price"].shift(3)) ** (1/3) - 1.0
        df["price_mom_3m"] = df["price_mom_3m"].fillna(0.0)
        df["price_level_log_wa_dev"] = df["price_level_log"] - df.groupby("month")["price_level_log"].transform("mean")
        df["price_yield_wa_dev"] = df["price_yield"] - df.groupby("month")["price_yield"].transform("mean")
        if "price_allocation_weight_sum" in df.columns:
            df["price_allocation_weight_sum"] = df["price_allocation_weight_sum"].clip(lower=0.0, upper=1.2)
        if "price_suburb_count" in df.columns:
            df["price_suburb_count"] = df["price_suburb_count"].fillna(0.0)
    except Exception:
        pass

    # Momentum features and churn proxy
    df = add_rent_momentum(df, group_col="sa2_code", month_col="month", rent_col="median_rent")
    df = add_churn_proxy(df)

    # External signals (optional, month-level) + simple lags per SA2
    try:
        ext = pd.read_parquet(STAGE_DIR / "external_signals.parquet").copy()
        ext_cols = [c for c in ext.columns if c != "month"]
        if not ext.empty and ext_cols:
            ext["month"] = pd.to_datetime(ext["month"]).dt.to_period("M").dt.to_timestamp()
            df = df.merge(ext, on="month", how="left")
            df = df.sort_values(["sa2_code","month"]).copy()
            for c in ext_cols:
                df[f"{c}_lag1"] = df.groupby("sa2_code")[c].shift(1)
                df[f"{c}_lag3"] = df.groupby("sa2_code")[c].shift(3)
    except Exception:
        pass

    # Neighbor availability (optional)
    try:
        adj = pd.read_parquet(STAGE_DIR / "sa2_neighbors.parquet").copy()
        adj["sa2_code"] = adj["sa2_code"].astype(str)
        adj["neighbor"] = adj["neighbor"].astype(str)
        # Build neighbor mean for each (sa2, month)
        months = df["month"].drop_duplicates().to_frame().assign(_k=1)
        adj_rep = adj.assign(_k=1)
        adj_m = adj_rep.merge(months, on="_k").drop(columns=["_k"]).copy()
        neigh = (
            adj_m.merge(
                df[["sa2_code","month","availability_rate"]].rename(columns={"sa2_code":"neighbor","availability_rate":"_nbr"}),
                on=["neighbor","month"], how="left"
            )
            .groupby(["sa2_code","month"], as_index=False)["_nbr"].mean()
            .rename(columns={"_nbr": "nbr_availability_rate"})
        )
        df = df.merge(neigh, on=["sa2_code","month"], how="left")
    except Exception:
        pass

    # Label: next-month rent rise > threshold
    df["median_rent_next"] = df.groupby("sa2_code")["median_rent"].shift(-1)
    valid_label = df["median_rent_next"].notna() & df["median_rent"].notna()
    df["y"] = np.where(
        valid_label,
        ((df["median_rent_next"] - df["median_rent"]) / df["median_rent"] > threshold).astype(float),
        np.nan,
    )

    # Clip extreme feature values to stabilize sampling (robust to outliers)
    # Churn is a rate; cap to [0, 1]
    if "churn_rate" in df.columns:
        df["churn_rate"] = _winsorize_series(df["churn_rate"], lower_bound=0.0, upper_bound=1.0)
    # Momentum: quantile-based winsorisation with loose physical bounds
    if "rent_mom_1m" in df.columns:
        df["rent_mom_1m"] = _winsorize_series(df["rent_mom_1m"], lower_bound=-1.0, upper_bound=1.0)
    if "rent_mom_3m" in df.columns:
        df["rent_mom_3m"] = _winsorize_series(df["rent_mom_3m"], lower_bound=-0.8, upper_bound=0.8)

    # Calendar + simple lags
    df = add_calendar_features(df)
    df = df.sort_values(["sa2_code","month"]).copy()
    df["availability_rate_lag1"] = df.groupby("sa2_code")["availability_rate"].shift(1)
    df["churn_rate_lag1"] = df.groupby("sa2_code")["churn_rate"].shift(1)
    df["rent_mom_12m"] = (df["median_rent"] / df.groupby("sa2_code")["median_rent"].shift(12)) - 1.0
    df["rent_mom_12m"] = _winsorize_series(df["rent_mom_12m"], lower_bound=-1.5, upper_bound=1.5)

    # Rent shock features for low-supply regions
    df = add_time_since_spike(df, threshold=max(float(threshold), 0.02))
    if "time_since_spike" in df.columns:
        df["time_since_spike"] = df["time_since_spike"].clip(lower=0.0, upper=60.0)
    df = add_supply_shock_features(
        df,
        rent_spike_threshold=max(float(threshold), 0.02),
        availability_col="availability_rate",
        stock_col="stock_bonds",
    )
    if "rent_spike_magnitude" in df.columns:
        df["rent_spike_magnitude"] = _winsorize_series(df["rent_spike_magnitude"], lower_bound=0.0, upper_bound=1.0)
    df = add_sparse_market_features(
        df,
        rent_col="median_rent",
        lodgement_col="count_lodgements",
        stock_col="stock_bonds",
    )

    # Lodgement-based weights to down-weight noisy months (low coverage)
    lodge_k = float(os.getenv("LODGE_SMOOTHING", "12"))
    lodge_floor = float(os.getenv("LODGE_WEIGHT_FLOOR", "0.05"))
    df["lodgement_weight"] = compute_lodgement_weights(
        df,
        lodgements_col="count_lodgements",
        smoothing=lodge_k,
        floor=lodge_floor,
    )

    # Month-level demeaned variants for WA context
    demean_cols = ["availability_rate", "churn_rate", "rent_mom_1m", "rent_mom_3m"]
    df = add_group_demean(df, ["month"], demean_cols, suffix="_wa_dev")

    # Key feature interactions
    interaction_pairs = [
        ("availability_rate", "rent_mom_1m"),
        ("availability_rate", "churn_rate"),
        ("churn_rate", "rent_mom_3m"),
    ]
    df = add_interaction_features(df, interaction_pairs)

    # Features
    feat_cols = [
        "availability_rate", "churn_rate", "rent_mom_1m", "rent_mom_3m",
        "availability_rate_lag1", "churn_rate_lag1", "rent_mom_12m",
        "mo_sin","mo_cos","quarter","is_eofy","is_uni_sem_feb","is_uni_sem_jul",
    ]
    # Add price features if present
    for c in [
        "price_level_log","price_band_high","price_yield",
        "price_level_log_wa_dev","price_yield_wa_dev","price_mom_3m",
        "price_allocation_weight_sum","price_suburb_count",
    ]:
        if c in df.columns:
            feat_cols.append(c)
    # Price interactions
    if "price_band_high" in df.columns:
        df["availability_rate__x__price_band"] = df["availability_rate"] * df["price_band_high"]
        df["rent_mom_1m__x__price_band"] = df["rent_mom_1m"] * df["price_band_high"]
        for c in ["availability_rate__x__price_band","rent_mom_1m__x__price_band"]:
            feat_cols.append(c)
    for col in demean_cols:
        name = f"{col}_wa_dev"
        if name in df.columns:
            feat_cols.append(name)
    for a, b in interaction_pairs:
        name = f"{a}__x__{b}"
        if name in df.columns:
            feat_cols.append(name)
    for c in [
        "time_since_spike",
        "rent_spike_flag",
        "rent_spike_magnitude",
        "low_availability_flag",
        "rent_spike_low_availability",
        "stock_pressure_6m",
        "stock_crunch_flag",
        "rent_spike_stock_crunch",
        "lodgement_rel_3m",
        "lodgement_rel_12m",
        "lodgement_wa_ratio",
        "lodgement_wa_zscore",
        "low_lodgement_flag",
        "lodgement_dropout_flag",
        "rent_spike_recent_3m",
        "rent_spike_recent_6m",
        "rent_spike_ewma",
        "low_stock_flag_sparse",
        "thin_market_flag",
        "thin_market_spike_recent",
        "thin_market_spike_ewma",
        "thin_market_dropout",
    ]:
        if c in df.columns:
            feat_cols.append(c)
    if "lodgement_weight" in df.columns:
        feat_cols.append("lodgement_weight")
    # Optional ext + neighbor columns
    for c in ["nbr_availability_rate", "rba_cash_rate", "wa_unemp_rate", "wa_unemp_rate_sa",
              "wa_building_approvals", "wa_build_approvals_num",
              "rba_cash_rate_lag1","rba_cash_rate_lag3",
              "wa_unemp_rate_lag1","wa_unemp_rate_lag3",
              "wa_unemp_rate_sa_lag1","wa_unemp_rate_sa_lag3",
              "wa_building_approvals_lag1","wa_building_approvals_lag3",
              "wa_build_approvals_num_lag1","wa_build_approvals_num_lag3"]:
        if c in df.columns:
            feat_cols.append(c)
    # Drop features that are entirely NA to avoid NaN mean/std warnings and degenerate columns
    all_na_cols = [c for c in feat_cols if c in df.columns and df[c].notna().sum() == 0]
    if all_na_cols:
        for c in all_na_cols:
            df.drop(columns=[c], inplace=True, errors="ignore")
        feat_cols = [c for c in feat_cols if c not in all_na_cols]

    # Shrink volatility-driven features toward zero when coverage is thin
    if "lodgement_weight" in df.columns:
        shrink_targets = [
            "rent_mom_1m",
            "rent_mom_3m",
            "rent_mom_12m",
            "rent_spike_flag",
            "rent_spike_magnitude",
            "rent_spike_low_availability",
            "rent_spike_stock_crunch",
            "rent_spike_recent_3m",
            "rent_spike_recent_6m",
            "rent_spike_ewma",
            "thin_market_spike_recent",
            "thin_market_spike_ewma",
        ]
        for col in shrink_targets:
            if col in df.columns:
                df[col] = df[col] * df["lodgement_weight"]

    # Do not drop rows for feature NaNs — standardize_columns will impute
    # with feature-wise means. Only require the label to be present.
    df_model = df.dropna(subset=["y"]).copy()
    df_model["_is_pseudo"] = False
    grp = df_model.groupby("sa2_code")["y"].agg(["count","sum"]).reset_index()
    pseudo_rows: list[pd.Series] = []
    pseudo_weight = float(os.getenv("PSEUDO_LABEL_WEIGHT", "0.05"))
    for _, row in grp.iterrows():
        total = int(row["count"])
        positive = float(row["sum"])
        if total == 0:
            continue
        if positive == 0.0 or positive == float(total):
            sa2_code = str(row["sa2_code"])
            group_df = df_model[df_model["sa2_code"] == sa2_code]
            if group_df.empty:
                continue
            template = group_df.iloc[-1].copy()
            template["_is_pseudo"] = True
            template["y"] = 1.0 if positive == 0.0 else 0.0
            pseudo_rows.append(template)
    if pseudo_rows:
        pseudo_df = pd.DataFrame(pseudo_rows)
        df_model = pd.concat([df_model, pseudo_df], ignore_index=True)
    pseudo_mask = df_model["_is_pseudo"].astype(bool).to_numpy()

    # Gradient-boosting booster to assist thin markets (optional)
    booster_enabled = os.getenv("SPARSE_MARKET_BOOSTER", "1").lower() in {"1", "true", "yes"}
    booster_features = [c for c in feat_cols if c in df_model.columns]
    booster_model = None
    booster_means: dict[str, float] | None = None
    if booster_enabled and booster_features and len(df_model) >= 30:
        try:
            X_train = df_model[booster_features].copy()
            booster_means = {c: float(X_train[c].mean(skipna=True) or 0.0) for c in booster_features}
            for c, mean_val in booster_means.items():
                X_train[c] = X_train[c].fillna(mean_val)
            y_train = df_model["y"].astype(int).to_numpy()
            weight = np.ones(len(df_model), dtype=float)
            if "lodgement_weight" in df_model.columns:
                weight *= df_model["lodgement_weight"].fillna(1.0).to_numpy()
            if pseudo_mask.any():
                pseudo_scale = float(os.getenv("SPARSE_BOOSTER_PSEUDO_WEIGHT", "0.2"))
                weight[pseudo_mask] = pseudo_scale

            booster_model = GradientBoostingClassifier(
                n_estimators=int(os.getenv("SPARSE_BOOST_N_EST", "300")),
                learning_rate=float(os.getenv("SPARSE_BOOST_LR", "0.05")),
                subsample=float(os.getenv("SPARSE_BOOST_SUBSAMPLE", "0.7")),
                max_depth=int(os.getenv("SPARSE_BOOST_MAX_DEPTH", "3")),
                min_samples_leaf=int(os.getenv("SPARSE_BOOST_MIN_LEAF", "15")),
                random_state=RANDOM_SEED,
            )
            booster_model.fit(X_train.values, y_train, sample_weight=weight)

            prob_train = booster_model.predict_proba(X_train.values)[:, 1]
            df_model["booster_logit"] = _safe_logit(prob_train)

            X_full = df[booster_features].copy()
            for c, mean_val in booster_means.items():
                if c in X_full.columns:
                    X_full[c] = X_full[c].fillna(mean_val)
            prob_full = booster_model.predict_proba(X_full.values)[:, 1]
            df["booster_logit"] = _safe_logit(prob_full)
            if "booster_logit" not in feat_cols:
                feat_cols.append("booster_logit")
        except Exception as exc:
            print(f"[warn] sparse booster disabled: {exc}")
            booster_model = None

    df_model = df_model.drop(columns=["_is_pseudo"])

    # Observation weights based on lodgement coverage (post pseudo-injection)
    lodgement_weight = None
    if "lodgement_weight" in df_model.columns:
        lodgement_weight = df_model["lodgement_weight"].astype(float).to_numpy()
        min_w = float(os.getenv("LODGE_WEIGHT_FLOOR", "0.05"))
        lodgement_weight = np.clip(lodgement_weight, min_w, 1.0)

    # Standardize features
    Xz, mu, sd = standardize_columns(df_model, feat_cols)
    # Quantile-based clipping of standardized features to stabilise logits
    Xz = _clip_standardized_matrix(Xz)

    # Indices
    sa2_codes = pd.Index(sorted(df_model["sa2_code"].unique()))
    sa2_idx = pd.Categorical(df_model["sa2_code"], categories=sa2_codes).codes.astype(int)
    month_codes = pd.Index(sorted(df_model["month"].unique()))
    month_idx = pd.Categorical(df_model["month"], categories=month_codes).codes.astype(int)

    y = df_model["y"].astype(int).to_numpy()
    return (
        df_model,
        Xz,
        y,
        sa2_idx,
        sa2_codes,
        feat_cols,
        mu,
        sd,
        df,
        month_idx,
        month_codes,
        pseudo_mask,
        float(pseudo_weight),
        lodgement_weight,
    )

def fit_forecast(
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        recency_half_life: float | None = FORECAST_RECENCY_HALFLIFE,
        bias_correct_l6: bool = False,
    calibrate_isotonic: bool = False,
    calibrate_use_raw: bool = False,
        prior_shift: bool = False,
        auto_calibrate: bool = True,
        target_accept: float = 0.99,
        sampler: str = "pymc",
        init: str = "adapt_diag_grad",
        max_treedepth: int = 15,
        save_trace: str | None = None,
        summary_file: str | None = None,
        summary_vars: list[str] | None = None,
        print_summary: bool = False,
        threshold: float = RENT_GROWTH_THRESHOLD,
        output_suffix: str | None = None,
        rhat_max: float = 1.01,
        sampler_retries: int = 2,
        retry_draw_multiplier: float = 1.5,
    ):
    """Fit the hierarchical logistic forecast and write predictions + history."""
    (
        df_model,
        Xz,
        y,
        sa2_idx,
        sa2_codes,
        feat_cols,
        mu,
        sd,
        df_full,
        month_idx_train,
        month_codes,
        pseudo_mask,
        pseudo_weight,
        lodgement_weight,
    ) = build_dataset(threshold=threshold)
    n_sa2 = len(sa2_codes)
    n_month = len(month_codes)
    n, k = Xz.shape

    # Segment SA2s into price-based clusters (quin-tiles where possible)
    cluster_idx_train = np.zeros(len(df_model), dtype=int)
    cluster_map: dict[str, int] = {code: 0 for code in sa2_codes.astype(str)}
    price_col = None
    for candidate in ["price_level_log", "median_house_price", "price_yield"]:
        if candidate in df_model.columns and df_model[candidate].notna().any():
            price_col = candidate
            break
    if price_col is not None:
        sa2_price = (df_model[["sa2_code", price_col]]
                        .dropna()
                        .groupby("sa2_code")[price_col]
                        .mean())
        if not sa2_price.empty:
            unique_vals = sa2_price.dropna().unique()
            q = int(min(5, len(unique_vals)))
            if q > 1:
                labels = pd.qcut(sa2_price, q=q, labels=False, duplicates="drop")
            else:
                labels = sa2_price.map(lambda _: 0)
            for code, lab in labels.items():
                cluster_map[str(code)] = int(lab)
            cluster_idx_train = df_model["sa2_code"].map(lambda c: cluster_map.get(str(c), 0)).astype(int).to_numpy()
    n_cluster = int(max(cluster_map.values(), default=0) + 1)

    # Indices for key features to enable mild varying slopes
    idx_avail = feat_cols.index("availability_rate") if "availability_rate" in feat_cols else None
    idx_churn = feat_cols.index("churn_rate") if "churn_rate" in feat_cols else None
    idx_price = feat_cols.index("price_band_high") if "price_band_high" in feat_cols else None
    idx_rm1 = feat_cols.index("rent_mom_1m") if "rent_mom_1m" in feat_cols else None

    # Optional recency weights (half-life in months)
    pseudo_mask = np.asarray(pseudo_mask, dtype=bool)
    w = compute_recency_weights(df_model["month"], recency_half_life)
    if lodgement_weight is not None:
        if w is None:
            w = lodgement_weight
        else:
            w = np.asarray(w, dtype=float) * lodgement_weight
    if pseudo_mask.any():
        base_w = np.ones(len(df_model), dtype=float) if w is None else np.asarray(w, dtype=float)
        pseudo_w = max(1e-3, float(pseudo_weight))
        base_w[pseudo_mask] = pseudo_w
        w = base_w
    elif w is not None:
        w = np.asarray(w, dtype=float)

    real_mask = ~pseudo_mask if pseudo_mask.size else np.ones(len(y), dtype=bool)
    y_real = y[real_mask]

    # Center intercept at the empirical base rate (logit scale; clipped)
    pi = float(y_real.mean()) if len(y_real) else 0.5
    pi = max(1e-4, min(1 - 1e-4, pi))
    mu_intercept = np.log(pi / (1.0 - pi))

    enable_varying = os.getenv("VARYING_SLOPES", "1").lower() in {"1", "true", "yes"}

    with pm.Model() as m:
        # Global slopes
        beta = pm.Normal("beta", 0.0, 0.5, shape=k)

        # Random intercepts with relaxed shrinkage
        mu_a = pm.Normal("mu_a", mu_intercept, 1.5)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)

        # Month-level random effect to absorb regime shifts
        sigma_month = pm.HalfNormal("sigma_month", 1.0)
        z_month = pm.Normal("z_month", 0.0, 1.0, shape=n_month)
        month_effect = pm.Deterministic("month_effect", z_month * sigma_month)

        # Mild hierarchical varying slopes for key drivers (centered at 0)
        b_av_sa2 = None
        if enable_varying and idx_avail is not None:
            mu_b_av = pm.Normal("mu_b_av", 0.0, 0.5)
            sigma_b_av = pm.HalfNormal("sigma_b_av", 0.7)
            z_b_av = pm.Normal("z_b_av", 0.0, 1.0, shape=n_sa2)
            b_av_sa2 = pm.Deterministic("b_av_sa2", mu_b_av + z_b_av * sigma_b_av)

        b_ch_sa2 = None
        if enable_varying and idx_churn is not None:
            mu_b_ch = pm.Normal("mu_b_ch", 0.0, 0.5)
            sigma_b_ch = pm.HalfNormal("sigma_b_ch", 0.7)
            z_b_ch = pm.Normal("z_b_ch", 0.0, 1.0, shape=n_sa2)
            b_ch_sa2 = pm.Deterministic("b_ch_sa2", mu_b_ch + z_b_ch * sigma_b_ch)

        b_pr_sa2 = None
        if enable_varying and idx_price is not None:
            mu_b_pr = pm.Normal("mu_b_pr", 0.0, 0.5)
            sigma_b_pr = pm.HalfNormal("sigma_b_pr", 0.7)
            z_b_pr = pm.Normal("z_b_pr", 0.0, 1.0, shape=n_sa2)
            b_pr_sa2 = pm.Deterministic("b_pr_sa2", mu_b_pr + z_b_pr * sigma_b_pr)

        b_rm1_sa2 = None
        if enable_varying and idx_rm1 is not None:
            mu_b_rm1 = pm.Normal("mu_b_rm1", 0.0, 0.5)
            sigma_b_rm1 = pm.HalfNormal("sigma_b_rm1", 0.7)
            z_b_rm1 = pm.Normal("z_b_rm1", 0.0, 1.0, shape=n_sa2)
            b_rm1_sa2 = pm.Deterministic("b_rm1_sa2", mu_b_rm1 + z_b_rm1 * sigma_b_rm1)

        eta = a_sa2[sa2_idx] + month_effect[month_idx_train] + pm.math.dot(Xz, beta)
        if idx_avail is not None:
            eta = eta + b_av_sa2[sa2_idx] * Xz[:, idx_avail]
        if idx_churn is not None:
            eta = eta + b_ch_sa2[sa2_idx] * Xz[:, idx_churn]
        if idx_price is not None:
            eta = eta + b_pr_sa2[sa2_idx] * Xz[:, idx_price]
        if idx_rm1 is not None:
            eta = eta + b_rm1_sa2[sa2_idx] * Xz[:, idx_rm1]

        # Cluster-level random effect (price segments)
        sigma_cluster = pm.HalfNormal("sigma_cluster", 1.0)
        z_cluster = pm.Normal("z_cluster", 0.0, 1.0, shape=n_cluster)
        cluster_effect = pm.Deterministic("cluster_effect", z_cluster * sigma_cluster)
        eta = eta + cluster_effect[cluster_idx_train]
        if w is None:
            pm.Bernoulli("y", logit_p=eta, observed=y)
        else:
            y_dist = pm.Bernoulli.dist(logit_p=eta)
            pm.Potential("weighted_loglik_binom", (w * pm.logp(y_dist, y)).sum())

        def _run_pymc(draws_now: int, tune_now: int, init_arg: str) -> az.InferenceData:
            t0_local = time.time()
            idata_local = sample_nuts(
                draws=draws_now,
                tune=tune_now,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                init=init_arg,
                random_seed=RANDOM_SEED,
                progressbar=True,
            )
            try:
                print(f"[time] PyMC sampling took {time.time()-t0_local:.1f}s")
            except Exception:
                pass
            return idata_local

        def _sample_once(draws_now: int, tune_now: int) -> az.InferenceData:
            if sampler.lower() in {"blackjax", "numpyro"}:
                import os as _os

                _os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
                try:
                    import pymc.sampling.jax as pmjax  # type: ignore

                    t0_local = time.time()
                    if sampler.lower() == "blackjax":
                        idata_local = pmjax.sample_blackjax_nuts(
                            draws=draws_now,
                            tune=tune_now,
                            chains=chains,
                            target_accept=target_accept,
                            random_seed=RANDOM_SEED,
                            postprocessing_backend="cpu",
                        )
                    else:
                        idata_local = pmjax.sample_numpyro_nuts(
                            draws=draws_now,
                            tune=tune_now,
                            chains=chains,
                            target_accept=target_accept,
                            random_seed=RANDOM_SEED,
                            postprocessing_backend="cpu",
                        )
                    try:
                        print(f"[time] JAX sampling + postprocessing took {time.time()-t0_local:.1f}s")
                    except Exception:
                        pass
                    return idata_local
                except Exception:
                    # Fallback to standard PyMC sampler on any import/runtime issue
                    pass

            try:
                return _run_pymc(draws_now, tune_now, init)
            except ValueError as exc:
                if "Unknown initializer" in str(exc):
                    return _run_pymc(draws_now, tune_now, "jitter+adapt_diag")
                raise

        draws_now = max(int(draws), 1)
        tune_now = max(int(tune), 0)
        attempts = 0
        max_attempts = max(0, int(sampler_retries)) + 1
        idata = None

        while True:
            attempts += 1
            idata = _sample_once(draws_now, tune_now)

            max_rhat = float("nan")
            if chains >= 2:
                try:
                    rhat_da = az.rhat(idata, method="rank")
                    if hasattr(rhat_da, "to_array"):
                        max_rhat = float(np.nanmax(rhat_da.to_array().values))
                    else:
                        max_rhat = float(np.nanmax(np.asarray(rhat_da)))
                except Exception:
                    max_rhat = float("nan")

            if not math.isfinite(max_rhat):
                break
            if max_rhat <= rhat_max:
                break
            if attempts >= max_attempts:
                print(f"[warn] Max rhat {max_rhat:.3f} > {rhat_max:.3f} after {attempts} attempts; proceeding with latest sample.")
                break

            grows = max(1.0, float(retry_draw_multiplier))
            draws_next = int(math.ceil(draws_now * grows))
            tune_next = int(math.ceil(tune_now * grows))
            if draws_next == draws_now:
                draws_next += 100
            if tune_next == tune_now:
                tune_next += 100
            print(f"[warn] Max rhat {max_rhat:.3f} > {rhat_max:.3f}; retrying with draws={draws_next}, tune={tune_next}.")
            draws_now, tune_now = draws_next, tune_next
        draws = draws_now
        tune = tune_now
        try:
            div = int(idata.sample_stats["diverging"].sum().item()) if "diverging" in idata.sample_stats else 0
            td_max = None
            try:
                td_max = int(np.asarray(idata.sample_stats.get("tree_depth", None)).max())  # type: ignore
            except Exception:
                td_max = None
            if div:
                print(f"[warn] NUTS reported {div} divergences; consider higher --target-accept (e.g., 0.99–0.995) or heavier priors.")
            if td_max is not None:
                print(f"[info] Max tree_depth observed: {td_max}")
        except Exception:
            pass

    # ---- Predict for next month using latest available features per SA2 ----
    latest_month = df_full["month"].max()
    base = (df_full[df_full["month"] == latest_month]
            .copy()
            .drop_duplicates(subset=["sa2_code"]))

    # Build standardized X for prediction (impute with training means if needed)
    Xp_z = apply_standardization(base, feat_cols, mu, sd)
    Xp_z = np.clip(Xp_z, -8.0, 8.0)

    # Align SA2 index with training set
    idx_map = {code: i for i, code in enumerate(sa2_codes)}
    base["sa2_idx"] = [idx_map.get(code, -1) for code in base["sa2_code"]]

    # === Extract posterior with explicit, stable axis order ===
    # beta: dims (chain, draw, k) -> stack to (sample, k)
    beta_da = idata.posterior["beta"].stack(sample=("chain", "draw"))
    k_dim = next(d for d in beta_da.dims if d != "sample")
    beta_s = beta_da.transpose("sample", k_dim).values  # (S, k)

    # a_sa2: dims (chain, draw, n_sa2) -> (sample, n_sa2)
    a_da = idata.posterior["a_sa2"].stack(sample=("chain", "draw"))
    i_dim = next(d for d in a_da.dims if d != "sample")
    a_sa2_s = a_da.transpose("sample", i_dim).values   # (S, n_sa2)

    # month_effect: dims (chain, draw, n_month) -> (sample, n_month)
    month_da = idata.posterior["month_effect"].stack(sample=("chain", "draw"))
    j_dim = next(d for d in month_da.dims if d != "sample")
    month_s = month_da.transpose("sample", j_dim).values  # (S, n_month)

    cluster_da = idata.posterior["cluster_effect"].stack(sample=("chain", "draw"))
    k_dim_cluster = next(d for d in cluster_da.dims if d != "sample")
    cluster_s = cluster_da.transpose("sample", k_dim_cluster).values  # (S, n_cluster)

    # Optional varying slopes arrays if present
    b_av_s = None
    b_ch_s = None
    try:
        b_av_da = idata.posterior["b_av_sa2"].stack(sample=("chain", "draw"))
        i_dim_ba = next(d for d in b_av_da.dims if d != "sample")
        b_av_s = b_av_da.transpose("sample", i_dim_ba).values  # (S, n_sa2)
    except Exception:
        b_av_s = None
    try:
        b_ch_da = idata.posterior["b_ch_sa2"].stack(sample=("chain", "draw"))
        i_dim_bc = next(d for d in b_ch_da.dims if d != "sample")
        b_ch_s = b_ch_da.transpose("sample", i_dim_bc).values  # (S, n_sa2)
    except Exception:
        b_ch_s = None

    S = beta_s.shape[0]
    month_index_map = {pd.Timestamp(m): i for i, m in enumerate(month_codes)}

    cluster_idx_train_arr = df_model["sa2_code"].map(lambda c: cluster_map.get(str(c), 0)).astype(int).to_numpy()

    def _posterior_summary(X_rows: np.ndarray,
                           sa2_indices: np.ndarray,
                           month_indices: np.ndarray,
                           cluster_indices: np.ndarray,
                           *, quantiles: bool = False):
        X_rows = np.asarray(X_rows)
        sa2_indices = np.asarray(sa2_indices, dtype=int)
        month_indices = np.asarray(month_indices, dtype=int)
        cluster_indices = np.asarray(cluster_indices, dtype=int)
        n_obs = len(sa2_indices)
        mean = np.zeros(n_obs, dtype=float)
        if quantiles:
            p05 = np.zeros(n_obs, dtype=float)
            p50 = np.zeros(n_obs, dtype=float)
            p95 = np.zeros(n_obs, dtype=float)
        for i in range(n_obs):
            idx = int(sa2_indices[i])
            lin = beta_s @ X_rows[i, :]
            if idx >= 0:
                lin = lin + a_sa2_s[:, idx]
                if (b_av_s is not None) and (idx_avail is not None):
                    lin = lin + b_av_s[:, idx] * float(X_rows[i, idx_avail])
                if (b_ch_s is not None) and (idx_churn is not None):
                    lin = lin + b_ch_s[:, idx] * float(X_rows[i, idx_churn])
            midx = int(month_indices[i])
            if midx >= 0:
                lin = lin + month_s[:, midx]
            cidx = int(cluster_indices[i])
            if cidx >= 0:
                lin = lin + cluster_s[:, cidx]
            ps = _expit(lin)
            mean[i] = ps.mean()
            if quantiles:
                p05[i] = np.quantile(ps, 0.05)
                p50[i] = np.quantile(ps, 0.50)
                p95[i] = np.quantile(ps, 0.95)
        if quantiles:
            return mean, p05, p50, p95
        return mean

    # Training-set posterior means for diagnostics / threshold tuning
    train_prob_mean = _posterior_summary(Xz, sa2_idx, month_idx_train, cluster_idx_train_arr, quantiles=False)

    # Forecast probabilities with credible intervals
    base_month_idx = base["month"].map(lambda v: month_index_map.get(pd.Timestamp(v), -1)).to_numpy()
    base_cluster_idx = base["sa2_code"].map(lambda v: cluster_map.get(str(v), 0)).to_numpy()
    base_probs, prob_p05, prob_p50, prob_p95 = _posterior_summary(
        Xp_z, base["sa2_idx"].to_numpy(), base_month_idx, base_cluster_idx, quantiles=True
    )

    out = pd.DataFrame({
        "sa2_code": base["sa2_code"].values,
        "month": (latest_month + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp(),
        "price_pressure_prob": base_probs,
        "prob_p05": prob_p05,
        "prob_p50": prob_p50,
        "prob_p95": prob_p95,
    })
    out["price_cluster"] = base_cluster_idx
    prior_shift_applied = False
    bias_correct_applied = False
    calibration_applied = False
    # Optional prior-shift (label-prevalence) adjustment of logits toward recent base rate
    if prior_shift or str(os.getenv("PRIOR_SHIFT", "0")).lower() in {"1","true","yes"}:
        try:
            # Training prevalence (exclude pseudo observations)
            pi_train = float(y_real.mean()) if len(y_real) else None
            # Estimate recent realized prevalence over last K months
            Kp = int(os.getenv("PRIOR_MONTHS", "3"))
            sa2_full = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
            sa2_full["month"] = pd.to_datetime(sa2_full["month"]).dt.to_period("M").dt.to_timestamp()
            sa2_full = sa2_full.sort_values(["sa2_code","month"]).copy()
            sa2_full["rent_prev"] = sa2_full.groupby("sa2_code")["median_rent"].shift(1)
            sa2_full["actual_jump"] = (sa2_full["median_rent"] / sa2_full["rent_prev"] - 1.0) > threshold
            realized_m = (sa2_full.loc[sa2_full["rent_prev"].notna()]
                                   .groupby("month", as_index=False)["actual_jump"].mean()
                                   .sort_values("month"))
            pi_test = float(realized_m.tail(max(Kp,1))["actual_jump"].mean()) if not realized_m.empty else None
            # Fallback to predicted mean if realized missing
            if not pi_test or not np.isfinite(pi_test) or pi_test <= 0 or pi_test >= 1:
                pi_test = float(np.clip(out["price_pressure_prob"].mean(), 1e-6, 1-1e-6))
            if pi_train and 0 < pi_train < 1 and 0 < pi_test < 1:
                logit = lambda a: np.log(a/(1-a))
                delta = logit(pi_test) - logit(pi_train)
                p_raw = np.clip(out["price_pressure_prob"].to_numpy(), 1e-6, 1-1e-6)
                p_adj = 1.0/(1.0 + np.exp(-(logit(p_raw) + delta)))
                out["price_pressure_prob_raw"] = out.get("price_pressure_prob_raw", out["price_pressure_prob"])  # keep if set by isotonic
                out["price_pressure_prob"] = p_adj
                prior_shift_applied = True
        except Exception:
            pass
    # Optional isotonic calibration using last K realized months
    env_calibrate = str(os.getenv("CALIBRATE_ISOTONIC", "0")).lower() in {"1","true","yes"}
    env_calib_use_raw = str(os.getenv("CALIB_USE_RAW", "0")).lower() in {"1","true","yes"}
    calib_input_use_raw = calibrate_use_raw or env_calib_use_raw
    should_calibrate = calibrate_isotonic or env_calibrate
    if auto_calibrate and not should_calibrate:
        if df_model["month"].nunique() >= 6 and df_model["y"].sum() >= 20:
            should_calibrate = True
    if should_calibrate:
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore
            # Build training pairs from history joined with realized labels
            hist_name_calib = "price_pressure_forecast_sa2_history.parquet" if not output_suffix else f"price_pressure_forecast_sa2_history_{output_suffix}.parquet"
            hist_path = STAGE_DIR / hist_name_calib
            hist = pd.read_parquet(hist_path)
            sa2_full = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
            sa2_full["month"] = pd.to_datetime(sa2_full["month"]).dt.to_period("M").dt.to_timestamp()
            sa2_full = sa2_full.sort_values(["sa2_code","month"]).copy()
            sa2_full["rent_prev"] = sa2_full.groupby("sa2_code")["median_rent"].shift(1)
            sa2_full["actual_jump"] = (sa2_full["median_rent"] / sa2_full["rent_prev"] - 1.0) > threshold
            realized = sa2_full.loc[sa2_full["rent_prev"].notna(), ["sa2_code","month","actual_jump"]]
            j = (hist.merge(realized, on=["sa2_code","month"], how="inner")
                    .sort_values(["month"]))
            Kc = int(os.getenv("CALIB_MONTHS", "6"))
            if not j.empty:
                # Use last Kc months where both pred and actual exist
                jm = j.groupby("month").size().reset_index(name="n")
                last_months = jm["month"].sort_values().tail(max(Kc,1)).tolist()
                cal = j[j["month"].isin(last_months)].copy()
                prob_col = "price_pressure_prob_raw" if (calib_input_use_raw and "price_pressure_prob_raw" in cal.columns) else "price_pressure_prob"
                p_tr = cal[prob_col].astype(float).to_numpy()
                y_tr = cal["actual_jump"].astype(int).to_numpy()
                if "price_pressure_prob_raw" not in out.columns:
                    out["price_pressure_prob_raw"] = out["price_pressure_prob"]
                base_probs = out["price_pressure_prob_raw"] if (calib_input_use_raw and "price_pressure_prob_raw" in out.columns) else out["price_pressure_prob"]
                applied_clusters = 0
                cluster_col = "price_cluster" if "price_cluster" in cal.columns else None
                if cluster_col and cal[cluster_col].notna().any() and cluster_col in out.columns:
                    for cluster_val, grp in cal.groupby(cluster_col):
                        grp = grp.dropna(subset=[prob_col])
                        p_tr_grp = grp[prob_col].astype(float).to_numpy()
                        y_tr_grp = grp["actual_jump"].astype(int).to_numpy()
                        if p_tr_grp.size < 10 or len(set(y_tr_grp)) < 2:
                            continue
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(p_tr_grp, y_tr_grp)
                        mask_out = out[cluster_col] == cluster_val
                        if mask_out.any():
                            out.loc[mask_out, "price_pressure_prob"] = iso.transform(base_probs.loc[mask_out].to_numpy())
                            applied_clusters += 1
                if applied_clusters == 0:
                    if p_tr.size >= 10 and len(set(y_tr)) == 2:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(p_tr, y_tr)
                        out["price_pressure_prob"] = iso.transform(base_probs.to_numpy())
                        calibration_applied = True
                else:
                    calibration_applied = True
        except Exception:
            pass
    # Optional per-SA2 bias correction using last 6 realized months
    if bias_correct_l6 or str(os.getenv("BIAS_CORRECT_L6", "0")).lower() in {"1","true","yes"}:
        try:
            # Load history (if any) and realized outcomes
            hist_name_bias = "price_pressure_forecast_sa2_history.parquet" if not output_suffix else f"price_pressure_forecast_sa2_history_{output_suffix}.parquet"
            hist_path = STAGE_DIR / hist_name_bias
            try:
                hist = pd.read_parquet(hist_path)
            except FileNotFoundError:
                hist = pd.DataFrame(columns=["sa2_code","month","price_pressure_prob"])
            sa2_full = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
            sa2_full["month"] = pd.to_datetime(sa2_full["month"]).dt.to_period("M").dt.to_timestamp()
            sa2_full = sa2_full.sort_values(["sa2_code","month"]).copy()
            sa2_full["rent_prev"] = sa2_full.groupby("sa2_code")["median_rent"].shift(1)
            # label using configured threshold
            sa2_full["actual_jump"] = (sa2_full["median_rent"] / sa2_full["rent_prev"] - 1.0) > threshold
            realized = sa2_full.loc[sa2_full["rent_prev"].notna(), ["sa2_code","month","actual_jump"]]
            # Join, take last K months per SA2
            joined = (hist.merge(realized, on=["sa2_code","month"], how="inner")
                           .sort_values(["sa2_code","month"]))
            K = int(os.getenv("BIAS_CORR_MONTHS", "6"))
            gamma = float(os.getenv("BIAS_CORR_GAMMA", "0.25"))
            # compute mean residual p - y over last K months per SA2
            joined["resid"] = joined["price_pressure_prob"].astype(float) - joined["actual_jump"].astype(int)
            if not joined.empty:
                joined_desc = joined.sort_values(["sa2_code", "month"], ascending=[True, False]).copy()
                joined_desc["_rank"] = joined_desc.groupby("sa2_code").cumcount()
                resid = (
                    joined_desc.loc[joined_desc["_rank"] < max(K, 1), ["sa2_code", "resid"]]
                    .groupby("sa2_code", as_index=False)["resid"].mean()
                    .rename(columns={"resid": "resid_mean"})
                )
            else:
                resid = pd.DataFrame(columns=["sa2_code", "resid_mean"])
            out = out.merge(resid, on="sa2_code", how="left")
            out["price_pressure_prob_raw"] = out["price_pressure_prob"]
            out["price_pressure_prob"] = np.clip(out["price_pressure_prob"] - gamma*out["resid_mean"].fillna(0.0), 0.0, 1.0)
            out = out.drop(columns=["resid_mean"])  # keep raw as separate column
            bias_correct_applied = True
        except Exception:
            # Fail safe: skip bias correction on any error
            pass
    # --- Training-set diagnostics & recommended threshold ---
    train_prob_real = train_prob_mean[real_mask]
    unique_real = np.unique(y_real) if len(y_real) else []
    metrics = {
        "train_rows": int(real_mask.sum()),
        "train_positive": int(y_real.sum()),
        "train_base_rate": float(y_real.mean()) if len(y_real) else float("nan"),
        "train_auc": float(roc_auc_score(y_real, train_prob_real)) if len(unique_real) > 1 else float("nan"),
        "train_average_precision": float(average_precision_score(y_real, train_prob_real)) if len(unique_real) > 1 else float("nan"),
        "pseudo_observations": int(pseudo_mask.sum()),
        "pseudo_weight": float(pseudo_weight if pseudo_mask.any() else 0.0),
    }
    metrics["rent_jump_threshold"] = float(threshold)

    threshold_grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    threshold_stats = []
    for thr in threshold_grid:
        preds = (train_prob_real >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (y_real == 1)))
        fp = int(np.sum((preds == 1) & (y_real == 0)))
        tn = int(np.sum((preds == 0) & (y_real == 0)))
        fn = int(np.sum((preds == 0) & (y_real == 1)))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        accuracy = (tp + tn) / max(len(y_real), 1)
        threshold_stats.append({
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    best_stats = max(
        threshold_stats,
        key=lambda d: (d["f1"], d["recall"], -abs(d["threshold"] - 0.5))
    ) if threshold_stats else None
    metrics["threshold_grid"] = threshold_stats
    if best_stats:
        metrics["recommended_threshold"] = float(best_stats["threshold"])
        metrics["recommended_metrics"] = best_stats

    metrics["calibration"] = {
        "requested": bool(calibrate_isotonic),
        "env_override": env_calibrate,
        "auto_enabled": bool(auto_calibrate),
        "applied": bool(calibration_applied),
    }
    metrics["prior_shift_applied"] = bool(prior_shift_applied)
    metrics["bias_correction_applied"] = bool(bias_correct_applied)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_filename = "forecast_training_metrics.json" if not output_suffix else f"forecast_training_metrics_{output_suffix}.json"
    metrics_path = EVAL_DIR / metrics_filename
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out["rent_jump_threshold"] = float(threshold)

    base_filename = "price_pressure_forecast_sa2.parquet" if not output_suffix else f"price_pressure_forecast_sa2_{output_suffix}.parquet"
    out.to_parquet(STAGE_DIR / base_filename, index=False)
    print(f"Wrote price-pressure forecast → data_stage/{base_filename}")
    
    # ---- maintain an append-only history for the website time slider ----
    hist_filename = "price_pressure_forecast_sa2_history.parquet" if not output_suffix else f"price_pressure_forecast_sa2_history_{output_suffix}.parquet"
    hist_path = STAGE_DIR / hist_filename
    try:
        old = pd.read_parquet(hist_path)
        combined = (pd.concat([old, out], ignore_index=True)
                    .drop_duplicates(subset=["sa2_code", "month", "rent_jump_threshold"])
                    .sort_values(["sa2_code", "month"]))
    except FileNotFoundError:
        combined = out.copy()

    combined.to_parquet(hist_path, index=False)
    print(f"Updated forecast history → {hist_path} (months={combined['month'].nunique()})")

    # Optional: persist trace to NetCDF
    if save_trace:
        try:
            trace_path = Path(save_trace)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            idata.to_netcdf(trace_path)
            print(f"Saved posterior trace → {trace_path}")
        except Exception as exc:
            print(f"[warn] Failed to save trace to {save_trace}: {exc}")

    # Optional: produce diagnostics summary (R-hat, ESS)
    if summary_file or print_summary:
        try:
            var_sel = summary_vars if summary_vars else None
            summary = az.summary(idata, var_names=var_sel, filter_vars="like", round_to=3)
            if summary_file:
                sum_path = Path(summary_file)
                sum_path.parent.mkdir(parents=True, exist_ok=True)
                summary.to_csv(sum_path)
                print(f"Wrote posterior diagnostics → {sum_path}")
            if print_summary:
                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    print(summary)
        except Exception as exc:
            print(f"[warn] Failed to compute diagnostics summary: {exc}")


    return idata

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit hierarchical logistic forecast and write predictions + history")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--recency-half-life", type=float, default=FORECAST_RECENCY_HALFLIFE,
                    help="Half-life in months for sample recency weights (default from config; set with --no-recency-weights to disable)")
    ap.add_argument("--no-recency-weights", action="store_true",
                    help="Disable recency weighting (treat all months equally).")
    ap.add_argument("--bias-correct-l6", action="store_true",
                    help="Apply per-SA2 bias correction using last 6 realized months (env BIAS_CORR_MONTHS to change window)")
    ap.add_argument("--calibrate-isotonic", action="store_true",
                    help="Apply isotonic calibration using last K realized months (env CALIB_MONTHS)")
    ap.add_argument("--calib-use-raw", action="store_true",
                    help="Train isotonic calibrator on raw model probabilities instead of previously calibrated values.")
    ap.add_argument("--prior-shift", action="store_true",
                    help="Adjust logits toward recent realized prevalence (env PRIOR_MONTHS; default 3)")
    ap.add_argument("--target-accept", type=float, default=0.92,
                    help="Target acceptance rate for the NUTS sampler (default 0.92).")
    ap.add_argument("--max-treedepth", type=int, default=15,
                    help="Maximum tree depth for the PyMC NUTS sampler (default 15). Ignored for JAX backends.")
    ap.add_argument("--no-auto-calibrate", dest="auto_calibrate", action="store_false",
                    help="Disable automatic isotonic calibration when sufficient history exists.")
    ap.set_defaults(auto_calibrate=True)
    ap.add_argument("--sampler", choices=["pymc", "blackjax", "numpyro"], default="pymc",
                    help="NUTS backend: standard PyMC, or JAX-based BlackJAX/NumPyro if installed.")
    ap.add_argument("--init", type=str, default="jitter+adapt_diag",
                    help="Initialization method for PyMC sampler (default 'jitter+adapt_diag').")
    ap.add_argument("--save-trace", type=str, default=None,
                    help="Optional NetCDF path to save posterior draws.")
    ap.add_argument("--summary-file", type=str, default=None,
                    help="Optional CSV path for ArviZ summary (R-hat, ESS).")
    ap.add_argument("--summary-vars", nargs="*", default=None,
                    help="Variable name patterns to include in the summary (defaults to all posterior vars).")
    ap.add_argument("--print-summary", action="store_true",
                    help="Print ArviZ summary to stdout after sampling.")
    ap.add_argument("--threshold", type=float, default=RENT_GROWTH_THRESHOLD,
                    help=f"Rent growth threshold for the binary label (default {RENT_GROWTH_THRESHOLD}).")
    ap.add_argument("--output-suffix", type=str, default=None,
                    help="Optional suffix appended to output files/metrics (e.g., 'thr002').")
    ap.add_argument("--rhat-max", type=float, default=1.01,
                    help="Maximum acceptable rank-based R-hat before retrying sampling (default 1.01).")
    ap.add_argument("--sampler-retries", type=int, default=2,
                    help="Additional sampling retries (with more draws) allowed when R-hat exceeds the limit.")
    ap.add_argument("--retry-draw-multiplier", type=float, default=1.5,
                    help="Multiplier applied to draws/tune for each retry when R-hat is too high (default 1.5).")
    args = ap.parse_args()
    fit_forecast(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        recency_half_life=None if args.no_recency_weights else args.recency_half_life,
        bias_correct_l6=args.bias_correct_l6,
        calibrate_isotonic=args.calibrate_isotonic,
        calibrate_use_raw=args.calib_use_raw,
        prior_shift=args.prior_shift,
        auto_calibrate=args.auto_calibrate,
        target_accept=args.target_accept,
        sampler=args.sampler,
        init=args.init,
        max_treedepth=args.max_treedepth,
        save_trace=args.save_trace,
        summary_file=args.summary_file,
        summary_vars=args.summary_vars,
        print_summary=args.print_summary,
        threshold=args.threshold,
        output_suffix=args.output_suffix,
        rhat_max=args.rhat_max,
        sampler_retries=args.sampler_retries,
        retry_draw_multiplier=args.retry_draw_multiplier,
    )
