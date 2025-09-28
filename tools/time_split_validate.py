"""
Time-split training and validation for the WA rental forecast.

Implements a strict split:
 - Fit the availability nowcast (Negative Binomial with offset) on a train window.
 - Use that nowcast to produce out-of-sample availability_rate for base months.
 - Fit the hierarchical logistic forecast on the train window using those features.
 - Predict probabilities for a validation window (months val_start..val_end),
   using base-month features (month T-1).

Writes predictions into data_stage/price_pressure_forecast_sa2_history.parquet
so that src.reporting.evaluate_forecasts can score them (and the static site can consume).

Usage:
  python -m tools.time_split_validate \
    --train-start 2025-01 --train-end 2025-04 \
    --val-start 2025-06 --val-end 2025-09
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.config import (
    STAGE_DIR,
    RENT_GROWTH_THRESHOLD,
    RANDOM_SEED,
    FORECAST_RECENCY_HALFLIFE,
)
from src.features.dates import (
    build_month_index,
    compute_recency_weights,
    parse_ym as _parse_month,
    month_range,
    prev_month,
    to_month,
)
from src.features.engineering import (
    add_calendar_features,
    add_churn_proxy,
    add_group_demean,
    add_interaction_features,
    add_rent_momentum,
    apply_standardization,
    compute_wa_aggregates,
    ensure_columns,
    standardize_columns,
)
from src.models.nowcast_design import prepare_design

# Optional ML deps (graceful fallback)
try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # fallback below
try:
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
except Exception:  # pragma: no cover
    GradientBoostingClassifier = None  # type: ignore



def _add_wa_aggregates(sa2: pd.DataFrame) -> pd.DataFrame:
    """Compatibility shim: delegate to shared helper to avoid divergence."""
    return compute_wa_aggregates(sa2)
def _load_external_signals():
    import pandas as pd
    path = STAGE_DIR / "external_signals.parquet"
    try:
        df = pd.read_parquet(path).copy()
        df["month"] = to_month(df["month"])  # normalize
        # keep only numeric external columns
        ext_cols = [c for c in df.columns if c != "month"]
        return df[["month", *ext_cols]], ext_cols
    except Exception:
        return None, []


# ----------------- availability helpers -----------------


def _load_cached_nowcast() -> pd.DataFrame | None:
    """Return cached availability rates if the parquet exists."""
    path = STAGE_DIR / "availability_nowcast_sa2.parquet"
    try:
        df = pd.read_parquet(path).copy()
    except FileNotFoundError:
        return None
    df["month"] = to_month(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df


def _slice_cached_nowcast(
    cache: pd.DataFrame | None,
    months: set[pd.Timestamp],
) -> tuple[pd.DataFrame | None, set[pd.Timestamp]]:
    """Return cache restricted to ``months`` when possible.

    Returns the sliced DataFrame and the set of months missing from the cache.
    """
    if cache is None or not len(cache):
        return None, months

    want = {to_month(m) for m in months}
    subset = cache[cache["month"].isin(want)].copy()
    have = set(subset["month"].unique())
    missing = want - have
    if missing:
        return None, missing
    return subset, set()


# ----------------- small helpers -----------------


def parse_ym(s: str) -> pd.Timestamp:
    """argparse-compatible YYYY-MM parser."""
    try:
        return _parse_month(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


# ----------------- Nowcast (train-only fit; OOS predict) -----------------

@dataclass
class NowcastPosterior:
    # Posterior draws stacked (sample dimension = chain x draw)
    mu_a: np.ndarray            # (S,)
    beta_t: np.ndarray          # (S,)
    season_eff: np.ndarray      # (S, 12)
    a_sa2: np.ndarray           # (S, n_sa2_train)
    sa2_index: dict             # code -> idx in a_sa2
    months_index: dict          # Timestamp -> t integer index


def fit_nowcast_train(df: pd.DataFrame,
                      train_months: set[pd.Timestamp],
                      *, draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4,
                      target_accept: float = 0.95,
                      trace_dir: 'Path | None' = None,
                      trace_name: str | None = None,
                      recency_half_life: float | None = None) -> Tuple[NowcastPosterior, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Fit the NB nowcast on train_months only. Returns:
      - NowcastPosterior (stacked draws of components)
      - mu_mean_train (per-observation posterior mean mu for train rows)
      - log_stock_train
      - sa2_cat_train (Categorical codes)
    """
    df_work = df.copy()
    df_work["month"] = to_month(df_work["month"])
    df_work["sa2_code"] = df_work["sa2_code"].astype(str)
    ensure_columns(df_work, ["count_disposals", "stock_bonds"])  # response/exposure

    months_index = build_month_index(df_work["month"].unique())
    train_df = df_work[df_work["month"].isin(train_months)].copy()
    design = prepare_design(train_df, month_index=months_index)

    y = design.y
    log_stock = design.log_stock
    t = design.time_index
    season = design.season_index
    sa2_idx = design.sa2_index
    sa2_codes = design.sa2_codes
    n_sa2 = design.n_sa2

    # Optional recency weights (half-life in months; newer rows get higher weight)
    w_now = compute_recency_weights(design.frame["month"], recency_half_life)

    with pm.Model() as m:
        mu_a = pm.Normal("mu_a", 0.0, 1.0)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)

        beta_t = pm.Normal("beta_t", 0.0, 0.2)

        sigma_season = pm.HalfNormal("sigma_season", 0.5)
        season_raw = pm.Normal("season_raw", 0.0, 1.0, shape=12)
        season_eff = pm.Deterministic("season_eff", season_raw - pm.math.mean(season_raw))

        alpha_nb = pm.HalfNormal("alpha_nb", 1.5)

        eta = a_sa2[sa2_idx] + beta_t * t + season_eff[season] + log_stock
        mu = pm.Deterministic("mu", pm.math.exp(eta))

        if w_now is None:
            pm.NegativeBinomial("y", mu=mu, alpha=alpha_nb, observed=y)
        else:
            # Weighted log-likelihood via Potential
            y_dist = pm.NegativeBinomial.dist(mu=mu, alpha=alpha_nb)
            pm.Potential("weighted_loglik_nb", (w_now * pm.logp(y_dist, y)).sum())

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept,
            random_seed=RANDOM_SEED, progressbar=True
        )
    # Optional: persist trace for verification
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        name = trace_name or "nowcast.nc"
        try:
            idata.to_netcdf(trace_dir / name)
        except Exception:
            pass

    # Extract posterior for components
    def stack(name):
        da = idata.posterior[name].stack(sample=("chain", "draw"))
        # transpose sample to axis 0, and keep other dims order
        other_dims = [d for d in da.dims if d != "sample"]
        return da.transpose("sample", *other_dims).values

    # Scalars become (S,) after stacking; ensure 1-D
    mu_a_s = np.asarray(stack("mu_a")).reshape(-1)
    beta_t_s = np.asarray(stack("beta_t")).reshape(-1)
    season_eff_s = stack("season_eff")  # (S, 12)
    a_sa2_s = stack("a_sa2")            # (S, n_sa2)

    # Posterior mean mu for train observations (same as in src/model_nowcast)
    mu_mean = idata.posterior["mu"].mean(dim=("chain", "draw")).values.squeeze()

    post = NowcastPosterior(
        mu_a=mu_a_s,
        beta_t=beta_t_s,
        season_eff=season_eff_s,
        a_sa2=a_sa2_s,
        sa2_index={code: i for i, code in enumerate(sa2_codes)},
        months_index=design.month_index,
    )
    # Return keys for alignment (same order as y/log_stock/mu_mean)
    keys = design.frame[["sa2_code", "month"]].reset_index(drop=True)
    return post, mu_mean, log_stock, keys


def predict_availability(post: NowcastPosterior,
                         rows: pd.DataFrame) -> pd.Series:
    """
    Predict availability_rate for provided rows (sa2_code, month, stock_bonds).
    For SA2s unseen in training, use mu_a (population mean) as the intercept.
    Returns a Series aligned with rows.index.
    """
    S = post.mu_a.shape[0]
    beta_t = post.beta_t.reshape(S)
    mu_a = post.mu_a.reshape(S)
    season_eff = post.season_eff  # (S, 12)
    a_sa2 = post.a_sa2            # (S, n_sa2)

    # Precompute t and season per row
    t_vals = rows["month"].map(post.months_index).astype(int).to_numpy()
    season_idx = rows["month"].dt.month.to_numpy() - 1
    log_stock = np.log(rows["stock_bonds"].clip(lower=1.0).to_numpy())

    out = np.empty(len(rows), dtype=float)
    for i, (code, t_i, s_i, log_s) in enumerate(zip(rows["sa2_code"].astype(str).tolist(), t_vals, season_idx, log_stock)):
        idx = post.sa2_index.get(code, -1)
        a = a_sa2[:, idx] if idx >= 0 else mu_a  # unseen SA2 → population mean
        eta = a + beta_t * float(t_i) + season_eff[:, int(s_i)] + float(log_s)
        mu = np.exp(eta)
        out[i] = float(mu.mean() / np.exp(log_s))  # availability per stock
    return pd.Series(out, index=rows.index)


# ----------------- Forecast (train-only fit; predict val) -----------------

@dataclass
class ForecastPosterior:
    beta: np.ndarray       # (S, k)
    a_sa2: np.ndarray      # (S, n_sa2_train)
    month_effect: np.ndarray  # (S, n_months)
    cluster_effect: np.ndarray  # (S, n_clusters)
    sa2_index: dict        # code -> idx
    month_index: dict      # month Timestamp -> idx
    cluster_index: dict    # sa2_code -> cluster idx
    feat_mu: np.ndarray    # (k,) training standardization mean
    feat_sd: np.ndarray    # (k,) training standardization std
    divergences: int = 0


def fit_forecast_train(train_df: pd.DataFrame,
                       feat_cols: list[str], *, draws: int = 1000, tune: int = 1000,
                       chains: int = 4, cores: int = 4, target_accept: float = 0.99,
                       trace_dir: 'Path | None' = None,
                       trace_name: str | None = None,
                       leakage_canary: bool = False,
                       recency_half_life: float | None = None,
                       init: str = "adapt_diag_grad",
                       rhat_max: float = 1.01,
                       sampler_retries: int = 2,
                       retry_draw_multiplier: float = 1.5) -> Tuple[ForecastPosterior, pd.DataFrame]:
    # Prepare features/labels (impute NaNs with feature-wise train means to keep early months)
    df = train_df.dropna(subset=["y"]).copy()
    df["_is_pseudo"] = False
    grp = df.groupby("sa2_code")["y"].agg(["count", "sum"]).reset_index()
    pseudo_rows: list[pd.Series] = []
    pseudo_weight = float(os.getenv("PSEUDO_LABEL_WEIGHT", "0.05"))
    for _, row in grp.iterrows():
        count = int(row["count"])
        positive = float(row["sum"])
        if count == 0:
            continue
        if positive == 0.0 or positive == float(count):
            sa2_code = str(row["sa2_code"])
            group_df = df[df["sa2_code"] == sa2_code]
            if group_df.empty:
                continue
            template = group_df.iloc[-1].copy()
            template["_is_pseudo"] = True
            template["y"] = 1.0 if positive == 0.0 else 0.0
            pseudo_rows.append(template)
    if pseudo_rows:
        pseudo_df = pd.DataFrame(pseudo_rows)
        df = pd.concat([df, pseudo_df], ignore_index=True)
    pseudo_mask = df["_is_pseudo"].astype(bool).to_numpy()
    df = df.drop(columns=["_is_pseudo"])

    Xz, mu, sd = standardize_columns(df, feat_cols)
    Xz = _clip_standardized_matrix(Xz)

    sa2_codes = pd.Index(sorted(df["sa2_code"].unique()))
    sa2_cat = pd.Categorical(df["sa2_code"], categories=sa2_codes)
    sa2_idx = sa2_cat.codes.astype(int)
    month_codes = pd.Index(sorted(df["month"].unique()))
    month_cat = pd.Categorical(df["month"], categories=month_codes)
    month_idx = month_cat.codes.astype(int)
    y = df["y"].astype(int).to_numpy()
    if leakage_canary:
        # Randomly permute labels to collapse any leakage-driven signal
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(y)

    # Optional recency weights (half-life in months; newer rows get higher weight)
    w = compute_recency_weights(df["month"], recency_half_life)
    if pseudo_mask.any():
        base_w = np.ones(len(df), dtype=float) if w is None else np.asarray(w, dtype=float)
        base_w[pseudo_mask] = max(1e-3, pseudo_weight)
        w = base_w
    elif w is not None:
        w = np.asarray(w, dtype=float)

    n_sa2 = len(sa2_codes)
    n_month = len(month_codes)
    n, k = Xz.shape

    cluster_map: dict[str, int] = {code: 0 for code in sa2_codes.astype(str)}
    cluster_idx = np.zeros(len(df), dtype=int)
    price_col = None
    for candidate in ["price_level_log", "median_house_price", "price_yield"]:
        if candidate in df.columns and df[candidate].notna().any():
            price_col = candidate
            break
    if price_col is not None:
        sa2_price = (df[["sa2_code", price_col]]
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
            cluster_idx = df["sa2_code"].map(lambda c: cluster_map.get(str(c), 0)).astype(int).to_numpy()
    n_cluster = int(max(cluster_map.values(), default=0) + 1)

    with pm.Model() as m:
        beta = pm.Normal("beta", 0.0, 0.5, shape=k)
        mu_a = pm.Normal("mu_a", 0.0, 1.5)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)
        sigma_month = pm.HalfNormal("sigma_month", 1.0)
        z_month = pm.Normal("z_month", 0.0, 1.0, shape=n_month)
        month_effect = pm.Deterministic("month_effect", z_month * sigma_month)
        sigma_cluster = pm.HalfNormal("sigma_cluster", 1.0)
        z_cluster = pm.Normal("z_cluster", 0.0, 1.0, shape=n_cluster)
        cluster_effect = pm.Deterministic("cluster_effect", z_cluster * sigma_cluster)
        eta = a_sa2[sa2_idx] + month_effect[month_idx] + cluster_effect[cluster_idx] + pm.math.dot(Xz, beta)
        # Optional varying slopes by SA2 for key drivers
        idxmap = {c:i for i,c in enumerate(feat_cols)}
        idx_price = idxmap.get("price_band_high")
        idx_rm1 = idxmap.get("rent_mom_1m")
        if idx_price is not None:
            mu_b_pr = pm.Normal("mu_b_pr", 0.0, 0.5)
            sigma_b_pr = pm.HalfNormal("sigma_b_pr", 0.7)
            z_b_pr = pm.Normal("z_b_pr", 0.0, 1.0, shape=n_sa2)
            b_pr_sa2 = pm.Deterministic("b_pr_sa2", mu_b_pr + z_b_pr * sigma_b_pr)
            eta = eta + b_pr_sa2[sa2_idx] * Xz[:, idx_price]
        if idx_rm1 is not None:
            mu_b_rm1 = pm.Normal("mu_b_rm1", 0.0, 0.5)
            sigma_b_rm1 = pm.HalfNormal("sigma_b_rm1", 0.7)
            z_b_rm1 = pm.Normal("z_b_rm1", 0.0, 1.0, shape=n_sa2)
            b_rm1_sa2 = pm.Deterministic("b_rm1_sa2", mu_b_rm1 + z_b_rm1 * sigma_b_rm1)
            eta = eta + b_rm1_sa2[sa2_idx] * Xz[:, idx_rm1]
        if w is None:
            pm.Bernoulli("y", logit_p=eta, observed=y)
        else:
            y_dist = pm.Bernoulli.dist(logit_p=eta)
            pm.Potential("weighted_loglik_binom", (w * pm.logp(y_dist, y)).sum())
        def _run_pymc(draws_now: int, tune_now: int, init_arg: str) -> az.InferenceData:
            t0_local = time.time()
            idata_local = pm.sample(
                draws=draws_now,
                tune=tune_now,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=RANDOM_SEED,
                progressbar=True,
                init=init_arg,
            )
            try:
                print(f"[time] PyMC sampling took {time.time()-t0_local:.1f}s")
            except Exception:
                pass
            return idata_local

        def _sample_once(draws_now: int, tune_now: int) -> az.InferenceData:
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

            if not math.isfinite(max_rhat) or max_rhat <= rhat_max:
                break
            if attempts >= max_attempts:
                print(
                    f"[warn] Max rhat {max_rhat:.3f} > {rhat_max:.3f} after {attempts} attempts; proceeding with latest sample.",
                    flush=True,
                )
                break

            grows = max(1.0, float(retry_draw_multiplier))
            draws_next = int(math.ceil(draws_now * grows))
            tune_next = int(math.ceil(tune_now * grows))
            if draws_next == draws_now:
                draws_next += 100
            if tune_next == tune_now:
                tune_next += 100
            print(
                f"[warn] Max rhat {max_rhat:.3f} > {rhat_max:.3f}; retrying with draws={draws_next}, tune={tune_next}.",
                flush=True,
            )
            draws_now, tune_now = draws_next, tune_next

        draws = draws_now
        tune = tune_now
    # Optional: persist trace for verification
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        name = trace_name or "forecast.nc"
        try:
            idata.to_netcdf(trace_dir / name)
        except Exception:
            pass

    # Extract posterior
    def stack(name):
        da = idata.posterior[name].stack(sample=("chain", "draw"))
        other_dims = [d for d in da.dims if d != "sample"]
        return da.transpose("sample", *other_dims).values

    beta_s = stack("beta")  # (S, k)
    # align dims: ensure second axis = k
    if beta_s.ndim == 1:
        beta_s = beta_s.reshape(-1, 1)
    a_sa2_s = stack("a_sa2")  # (S, n_sa2)
    month_effect_s = stack("month_effect")  # (S, n_month)
    cluster_effect_s = stack("cluster_effect")  # (S, n_cluster)

    divergences = int(idata.sample_stats["diverging"].sum().item())

    post = ForecastPosterior(
        beta=beta_s,
        a_sa2=a_sa2_s,
        month_effect=month_effect_s,
        cluster_effect=cluster_effect_s,
        sa2_index={code: i for i, code in enumerate(sa2_codes)},
        month_index={pd.Timestamp(m): i for i, m in enumerate(month_codes)},
        cluster_index={code: cluster_map.get(str(code), 0) for code in sa2_codes.astype(str)},
        feat_mu=mu,
        feat_sd=sd,
        divergences=divergences,
    )
    return post, df


def predict_forecast(post: ForecastPosterior,
                     base_df: pd.DataFrame,
                     feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict probabilities for base_df rows using the posterior.
    base_df must include columns feat_cols and 'sa2_code'.
    For SA2s unseen in training, apply fixed effects only (no random intercept),
    matching the behavior in src/model_forecast.py.
    """
    Xz = apply_standardization(base_df, feat_cols, post.feat_mu, post.feat_sd)
    Xz = _clip_standardized_matrix(Xz)

    S, k = post.beta.shape
    probs = np.zeros(len(base_df), dtype=float)
    p05 = np.zeros(len(base_df), dtype=float)
    p50 = np.zeros(len(base_df), dtype=float)
    p95 = np.zeros(len(base_df), dtype=float)
    for i, row in base_df.reset_index(drop=True).iterrows():
        idx = post.sa2_index.get(str(row["sa2_code"]), -1)
        lin = post.beta @ Xz[i, :]  # (S,)
        if idx >= 0:
            lin = lin + post.a_sa2[:, idx]
        month_val = pd.Timestamp(row["month"]) if "month" in row else None
        if month_val is not None:
            midx = post.month_index.get(month_val, None)
            if midx is not None:
                lin = lin + post.month_effect[:, midx]
        cidx = post.cluster_index.get(str(row["sa2_code"]), 0)
        lin = lin + post.cluster_effect[:, cidx]
        cidx = post.cluster_index.get(str(row["sa2_code"]), 0)
        lin = lin + post.cluster_effect[:, cidx]
        # numerically stable expit to avoid overflow warnings on extreme logits
        lin_c = np.clip(lin, -500, 500)
        ps = 1.0 / (1.0 + np.exp(-lin_c))
        probs[i] = ps.mean()
        p05[i] = np.quantile(ps, 0.05)
        p50[i] = np.quantile(ps, 0.50)
        p95[i] = np.quantile(ps, 0.95)
    return probs, p05, p50, p95


# ----------------- GBM helpers (optional) -----------------


@dataclass
class GBMModel:
    backend: str
    model: object
    best_iteration: int | None = None


def _train_gbm_model(train_df: pd.DataFrame,
                     feat_cols: list[str], *,
                     recency_half_life: float | None = None,
                     random_state: int = RANDOM_SEED,
                     learning_rate: float = 0.05,
                     num_boost_round: int = 400,
                     early_stopping_rounds: int | None = 50,
                     val_fraction: float = 0.2,
                     log_period: int | None = 50) -> GBMModel:
    """Train a boosted-tree classifier with optional chrono validation."""
    if train_df.empty:
        raise SystemExit("GBM training received an empty frame.")

    X = train_df[feat_cols].to_numpy(dtype=float)
    y = train_df["y"].astype(int).to_numpy()
    sample_weight = compute_recency_weights(train_df["month"], recency_half_life)

    if lgb is not None:
        params = dict(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=learning_rate,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_data_in_leaf=20,
            verbose=-1,
            metric=["binary_logloss", "auc"],
            seed=random_state,
        )

        dtrain = lgb.Dataset(X, label=y, weight=sample_weight)
        valid_sets = None
        train_rows_count = len(X)
        val_rows = 0
        val_month_subset: list[pd.Timestamp] = []
        callbacks = []

        if log_period and log_period > 0:
            callbacks.append(lgb.log_evaluation(period=log_period))

        if val_fraction and 0 < val_fraction < 1:
            # Chronologically split by month so validation is forward-looking
            months_sorted = sorted(train_df["month"].unique())
            split_idx = int(len(months_sorted) * (1 - val_fraction))
            if split_idx < 1:
                split_idx = len(months_sorted) - 1
            split_idx = max(1, min(split_idx, len(months_sorted) - 1))
            train_month_subset = set(months_sorted[:split_idx])
            valid_month_subset = set(months_sorted[split_idx:])
            train_mask = train_df["month"].isin(train_month_subset)
            valid_mask = train_df["month"].isin(valid_month_subset)

            X_train = X[train_mask.to_numpy()]
            y_train = y[train_mask.to_numpy()]
            w_train = sample_weight[train_mask.to_numpy()] if sample_weight is not None else None

            X_valid = X[valid_mask.to_numpy()]
            y_valid = y[valid_mask.to_numpy()]
            w_valid = sample_weight[valid_mask.to_numpy()] if sample_weight is not None else None

            if len(X_valid) and len(X_train):
                dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
                dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=dtrain)
                valid_sets = [dvalid]
                val_rows = len(X_valid)
                train_rows_count = len(X_train)
                val_month_subset = sorted(valid_month_subset)
            else:
                valid_sets = None

        if val_rows:
            print(
                f"[gbm] train_rows={train_rows_count} val_rows={val_rows} "
                f"val_months={[m.strftime('%Y-%m') for m in val_month_subset]} rounds={num_boost_round}",
                flush=True,
            )
        else:
            print(
                f"[gbm] train_rows={train_rows_count} val_rows=0 rounds={num_boost_round}",
                flush=True,
            )

        if valid_sets is not None and early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        best_iter = booster.best_iteration or booster.current_iteration()
        return GBMModel("lightgbm", booster, best_iteration=best_iter)

    if GradientBoostingClassifier is not None:
        clf = GradientBoostingClassifier(random_state=random_state)
        clf.fit(X, y, sample_weight=sample_weight)
        return GBMModel("sklearn", clf)

    raise SystemExit("No GBM backend available. Install lightgbm or scikit-learn.")


def _predict_gbm(model: GBMModel, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Predict probabilities for ``df`` using a fitted GBM model."""
    if df.empty:
        return np.empty(0, dtype=float)
    X = df[feat_cols].to_numpy(dtype=float)
    if model.backend == "lightgbm":
        num_iter = model.best_iteration or getattr(model.model, "best_iteration", None)
        return model.model.predict(X, num_iteration=num_iter)
    return model.model.predict_proba(X)[:, 1]


# ----------------- Orchestration -----------------

def _build_feature_frame(sa2: pd.DataFrame,
                         availability_df: pd.DataFrame,
                         threshold: float) -> pd.DataFrame:
    """Merge availability into SA2, compute churn and labels, and sort."""
    df = sa2.merge(availability_df[["sa2_code", "month", "availability_rate"]],
                   on=["sa2_code", "month"], how="left").copy()
    df = df.sort_values(["sa2_code", "month"]).copy()
    df["median_rent_next"] = df.groupby("sa2_code")["median_rent"].shift(-1)
    df["y"] = ((df["median_rent_next"] - df["median_rent"]) / df["median_rent"] > threshold).astype(float)
    df = add_churn_proxy(df)
    return df


def _assert_no_forbidden_features(feat_cols: list[str]) -> None:
    """Guardrail: ensure target-lookahead columns are not used as features."""
    forbidden = {"median_rent_next", "actual_jump", "y_next", "label_next"}
    bad = sorted(forbidden.intersection(set(feat_cols)))
    if bad:
        raise SystemExit(f"Leakage risk: forbidden future-looking columns in features: {bad}")


def _ensure_adjacency() -> 'pd.DataFrame | None':
    """Build or load SA2 adjacency list if available.
    Returns a DataFrame sa2_code, neighbor or None on failure.
    """
    from pathlib import Path as _Path
    path = STAGE_DIR / "sa2_neighbors.parquet"
    try:
        import pandas as _pd
        if path.exists():
            adj = _pd.read_parquet(path)
        else:
            from tools.spatial_features import build_sa2_adjacency
            adj = build_sa2_adjacency(path)
        adj["sa2_code"] = adj["sa2_code"].astype(str)
        adj["neighbor"] = adj["neighbor"].astype(str)
        return adj
    except Exception:
        return None


def _add_neighbor_availability(df: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Add neighbor mean availability_rate per (sa2_code, month).

    df must contain columns: sa2_code, month, availability_rate.
    Returns a copy with column 'nbr_availability_rate'.
    """
    if adj is None or adj.empty:
        return df
    d = df[["sa2_code", "month", "availability_rate"]].copy()
    # Repeat adjacency over months
    months = (
        d["month"].drop_duplicates().to_frame().assign(_k=1)
    )
    adj_rep = adj.copy().assign(_k=1)
    adj_m = adj_rep.merge(months, on="_k").drop(columns=["_k"]).copy()
    # Join neighbor availability at same month
    neigh = (
        adj_m.merge(
            d.rename(columns={"sa2_code": "neighbor", "availability_rate": "_nbr_avail"}),
            on=["neighbor", "month"], how="left"
        )
        .groupby(["sa2_code", "month"], as_index=False)["_nbr_avail"].mean()
        .rename(columns={"_nbr_avail": "nbr_availability_rate"})
    )
    out = df.merge(neigh, on=["sa2_code", "month"], how="left")
    return out


def _winsorize_series(s: pd.Series,
                      *,
                      lower_q: float = 0.01,
                      upper_q: float = 0.99,
                      lower_bound: float | None = None,
                      upper_bound: float | None = None) -> pd.Series:
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


def _format_threshold_suffix(threshold: float) -> str:
    """Return a filesystem-friendly suffix for a given threshold."""
    formatted = f"{threshold:.3f}".rstrip("0").rstrip(".")
    if not formatted:
        formatted = "0"
    return f"thr{formatted.replace('.', 'p')}"


def main(train_start: pd.Timestamp,
         train_end: pd.Timestamp,
         val_start: pd.Timestamp,
         val_end: pd.Timestamp,
         threshold: float = RENT_GROWTH_THRESHOLD,
         walk_forward: bool = False,
         draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4,
         trace_dir: 'Path | None' = None,
         with_external: bool = False,
         with_spatial: bool = False,
         leakage_canary: bool = False,
         extra_features: bool = False,
         recency_half_life: float | None = FORECAST_RECENCY_HALFLIFE,
         use_gbm: bool = False,
         stack_gbm: bool = False,
         calibrate_isotonic: bool = False,
         gbm_learning_rate: float = 0.05,
         gbm_rounds: int = 400,
         gbm_early_stopping: int | None = 50,
         gbm_val_fraction: float = 0.2,
         gbm_log_period: int | None = 50,
         pymc_target_accept: float = 0.99,
         sampler_rhat_max: float = 1.01,
         sampler_retries: int = 2,
         retry_draw_multiplier: float = 1.5,
         force_refit_nowcast: bool = False,
         output_suffix: str | None = None):
    # Load SA2 panel
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    ensure_columns(sa2, ["sa2_code", "month", "median_rent", "count_disposals", "stock_bonds"])
    sa2["sa2_code"] = sa2["sa2_code"].astype(str)
    sa2["month"] = to_month(sa2["month"])  # normalize type
    sa2 = add_rent_momentum(sa2)

    # Check coverage
    have_months = set(sa2["month"].unique())
    train_months = set(month_range(train_start, train_end))
    val_months = set(month_range(val_start, val_end))
    base_months = {prev_month(m) for m in val_months}
    needed = train_months | base_months
    missing = sorted(m for m in needed if m not in have_months)
    if missing:
        raise SystemExit(f"Missing required months in SA2 panel: {[m.strftime('%Y-%m') for m in missing]}")

    feat_cols = ["availability_rate", "churn_rate", "rent_mom_1m", "rent_mom_3m"]
    interaction_pairs = [
        ("availability_rate", "rent_mom_1m"),
        ("availability_rate", "churn_rate"),
        ("churn_rate", "rent_mom_3m"),
    ]
    demean_cols = ["availability_rate", "churn_rate", "rent_mom_1m", "rent_mom_3m"]

    new_df = None

    # Prepare optional adjacency once
    adj = _ensure_adjacency() if with_spatial else None

    # Optional cached availability to skip PyMC sampling when present
    cached_nowcast = None if force_refit_nowcast else _load_cached_nowcast()

    if not walk_forward:
        # ===== Fixed-split mode =====
        required_months = train_months | base_months
        avail_all, missing_cached = _slice_cached_nowcast(cached_nowcast, required_months)
        using_cached = avail_all is not None
        if using_cached:
            print(
                f"[nowcast] Using cached availability for {len(required_months)} months; "
                "skipping PyMC nowcast fit."
            )
        else:
            if cached_nowcast is not None and missing_cached:
                miss = ", ".join(sorted(m.strftime("%Y-%m") for m in sorted(missing_cached)))
                print(f"[nowcast] Cached availability missing months: {miss}. Refitting nowcast…")
            # Fit nowcast on train window only
            post_now, mu_mean_train, log_stock_train, train_keys = fit_nowcast_train(
                sa2, train_months, draws=draws, tune=tune, chains=chains, cores=cores,
                trace_dir=trace_dir,
                trace_name=f"nowcast_{train_start.strftime('%Y-%m')}_{train_end.strftime('%Y-%m')}.nc" if trace_dir else None,
                recency_half_life=recency_half_life,
            )

            # Availability for train (in-sample) and base months (OOS)
            avail_train = pd.DataFrame({
                "sa2_code": train_keys["sa2_code"].values,
                "month": train_keys["month"].values,
                "availability_rate": (mu_mean_train / np.exp(log_stock_train)),
            })
            base_rows = (sa2.loc[sa2["month"].isin(base_months), ["sa2_code", "month", "stock_bonds"]]
                           .copy()
                           .reset_index(drop=True))
            base_rows["availability_rate"] = predict_availability(post_now, base_rows)
            avail_all = pd.concat([avail_train, base_rows], ignore_index=True)

        # Spatial neighbor availability (safe: uses same-month availability)
        if with_spatial:
            avail_all = _add_neighbor_availability(avail_all, adj)

        df = _build_feature_frame(sa2, avail_all, threshold)
        # ---- House price features (monthly medians preferred; fallback to snapshot) ----
        try:
            hp_monthly_path = STAGE_DIR / "house_prices_sa2_monthly.parquet"
            hp_snapshot_path = STAGE_DIR / "house_prices_sa2_snapshot.parquet"
            hp: pd.DataFrame | None = None
            if hp_monthly_path.exists():
                hp = pd.read_parquet(hp_monthly_path).copy()
            elif hp_snapshot_path.exists():
                hp = pd.read_parquet(hp_snapshot_path).copy()
            if hp is not None and not hp.empty:
                rename_cols = {}
                if "allocation_weight_sum" in hp.columns:
                    rename_cols["allocation_weight_sum"] = "price_allocation_weight_sum"
                if "n_suburbs" in hp.columns:
                    rename_cols["n_suburbs"] = "price_suburb_count"
                if rename_cols:
                    hp = hp.rename(columns=rename_cols)
                keep_cols = [
                    c for c in ["sa2_code", "month", "median_house_price",
                                 "price_allocation_weight_sum", "price_suburb_count"]
                    if c in hp.columns
                ]
                hp = hp[keep_cols].copy()
                hp["sa2_code"] = hp["sa2_code"].astype(str)
                hp["month"] = to_month(hp["month"])  # normalize

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

                hp_exp = hp.groupby("sa2_code", group_keys=False).apply(_expand)
                df = df.merge(hp_exp, on=["sa2_code", "month"], how="left")
                df = df.sort_values(["sa2_code", "month"]).copy()

                df["price_level_log"] = np.log(df["median_house_price"])
                wa_med = df.groupby("month")["median_house_price"].transform("median")
                df["price_band_high"] = (df["median_house_price"] >= wa_med).astype(float)
                df["price_yield"] = (df["median_rent"] * 52.0) / df["median_house_price"]
                df["price_mom_3m"] = (df["median_house_price"] / df.groupby("sa2_code")["median_house_price"].shift(3)) ** (1/3) - 1.0
                df["price_mom_3m"] = df["price_mom_3m"].fillna(0.0)
                df["price_level_log_wa_dev"] = df["price_level_log"] - df.groupby("month")["price_level_log"].transform("mean")
                df["price_yield_wa_dev"] = df["price_yield"] - df.groupby("month")["price_yield"].transform("mean")
                if "price_allocation_weight_sum" in df.columns:
                    df["price_allocation_weight_sum"] = df["price_allocation_weight_sum"].clip(lower=0.0, upper=1.2)
                if "price_suburb_count" in df.columns:
                    df["price_suburb_count"] = df["price_suburb_count"].fillna(0.0)

                df["availability_rate__x__price_band"] = df["availability_rate"] * df["price_band_high"]
                df["rent_mom_1m__x__price_band"] = df["rent_mom_1m"] * df["price_band_high"]
                for c in [
                    "price_level_log", "price_band_high", "price_yield",
                    "price_level_log_wa_dev", "price_yield_wa_dev", "price_mom_3m",
                    "price_allocation_weight_sum", "price_suburb_count",
                    "availability_rate__x__price_band", "rent_mom_1m__x__price_band",
                ]:
                    if c not in feat_cols and c in df.columns:
                        feat_cols.append(c)
        except Exception:
            pass
        if with_spatial and "nbr_availability_rate" in avail_all.columns:
            # Merge neighbor feature into df
            df = df.merge(
                avail_all[["sa2_code", "month", "nbr_availability_rate"]].drop_duplicates(),
                on=["sa2_code", "month"], how="left")
            if "nbr_availability_rate" not in feat_cols:
                feat_cols.append("nbr_availability_rate")
        # === Merge external signals (optional) ===
        if with_external:
            ext_df, ext_cols = _load_external_signals()
            if ext_df is not None and not ext_df.empty:
                df = df.merge(ext_df, on="month", how="left")
                # Create simple lags per SA2 row (external series are WA-level, so group-by SA2 is safe)
                df = df.sort_values(["sa2_code","month"])
                new_cols=[]
                for c in ext_cols:
                    if c not in feat_cols:
                        feat_cols.append(c)
                    c1=f"{c}_lag1"; c3=f"{c}_lag3"
                    df[c1] = df.groupby("sa2_code")[c].shift(1)
                    df[c3] = df.groupby("sa2_code")[c].shift(3)
                    new_cols += [c1,c3]
                for nc in new_cols:
                    if nc not in feat_cols:
                        feat_cols.append(nc)
        # Calendar feats (sin/cos + flags)
        if extra_features:
            df = add_calendar_features(df)
            for c in ["mo_sin","mo_cos","quarter","is_eofy","is_uni_sem_feb","is_uni_sem_jul"]:
                if c not in feat_cols:
                    feat_cols.append(c)

        # Simple additional temporal features
        if extra_features:
            df = df.sort_values(["sa2_code","month"]).copy()
            df["availability_rate_lag1"] = df.groupby("sa2_code")["availability_rate"].shift(1)
            df["churn_rate_lag1"] = df.groupby("sa2_code")["churn_rate"].shift(1)
            df["rent_mom_12m"] = (df["median_rent"] / df.groupby("sa2_code")["median_rent"].shift(12)) - 1.0
            for c in ["availability_rate_lag1","churn_rate_lag1","rent_mom_12m"]:
                if c not in feat_cols:
                    feat_cols.append(c)
        # WA aggregates
        if extra_features:
            wa = _add_wa_aggregates(sa2)
            df = df.merge(wa, on="month", how="left")
        if extra_features:
            if extra_features:
                for c in ["wa_rent_mom","wa_churn_rate","wa_stock_growth","wa_disp_rate"]:
                    if c not in feat_cols:
                        feat_cols.append(c)

        # Consistent engineered features with production pipeline
        df = add_group_demean(df, ["month"], demean_cols, suffix="_wa_dev")
        df = add_interaction_features(df, interaction_pairs)
        for col in demean_cols:
            name = f"{col}_wa_dev"
            if name not in feat_cols and name in df.columns:
                feat_cols.append(name)
        for a, b in interaction_pairs:
            name = f"{a}__x__{b}"
            if name not in feat_cols and name in df.columns:
                feat_cols.append(name)

        # Train model on train window (either Bayesian logistic or GBM)
        train_df = df[df["month"].isin(train_months)].copy()
        _assert_no_forbidden_features(feat_cols)
        post_fc = None
        logistic_failed = False
        if not (use_gbm and not stack_gbm):
            post_fc, _ = fit_forecast_train(
                train_df, feat_cols, draws=draws, tune=tune, chains=chains, cores=cores,
                target_accept=pymc_target_accept,
                trace_dir=trace_dir,
                trace_name=(f"forecast_{train_start.strftime('%Y-%m')}_{train_end.strftime('%Y-%m')}"
                            f"_val_{val_start.strftime('%Y-%m')}_{val_end.strftime('%Y-%m')}.nc") if trace_dir else None,
                leakage_canary=leakage_canary,
                recency_half_life=recency_half_life,
                rhat_max=sampler_rhat_max,
                sampler_retries=sampler_retries,
                retry_draw_multiplier=retry_draw_multiplier,
            )
            if post_fc.divergences:
                print(
                    f"[warn] PyMC forecast sampler produced {post_fc.divergences} divergences; "
                    "skipping Bayesian logistic component.",
                    flush=True,
                )
                logistic_failed = True
                post_fc = None
        gbm_model = None
        if use_gbm or stack_gbm:
            gbm_model = _train_gbm_model(
                train_df, feat_cols,
                recency_half_life=recency_half_life,
                random_state=RANDOM_SEED,
                learning_rate=gbm_learning_rate,
                num_boost_round=gbm_rounds,
                early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                val_fraction=gbm_val_fraction,
                log_period=gbm_log_period,
            )

        iso = None
        if calibrate_isotonic:
            try:
                from sklearn.isotonic import IsotonicRegression  # type: ignore
                p_train = None
                if use_gbm or stack_gbm:
                    if gbm_model is None:
                        gbm_model = _train_gbm_model(
                            train_df, feat_cols,
                            recency_half_life=recency_half_life,
                            random_state=RANDOM_SEED,
                            learning_rate=gbm_learning_rate,
                            num_boost_round=gbm_rounds,
                            early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                            val_fraction=gbm_val_fraction,
                            log_period=gbm_log_period,
                        )
                    p_train = _predict_gbm(gbm_model, train_df, feat_cols)
                if p_train is None and post_fc is not None:
                    p_train, _, _, _ = predict_forecast(post_fc, train_df, feat_cols)
                y_train = train_df["y"].astype(int).to_numpy()
                if p_train is not None and len(p_train) and len(set(y_train)) > 1:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(p_train, y_train)
            except Exception:
                iso = None

        # Predict target months
        preds = []
        train_n = len(train_df)
        train_pos = int(train_df["y"].sum()) if "y" in train_df.columns else 0
        train_rate = train_pos / train_n if train_n else float("nan")
        for T in sorted(val_months):
            base_m = prev_month(T)
            base_df = df[df["month"] == base_m].copy()
            if base_df.empty:
                continue
            base_df = base_df.dropna(subset=["availability_rate"])  # ensure availability exists

            if post_fc is None:
                if gbm_model is None:
                    gbm_model = _train_gbm_model(
                        train_df, feat_cols,
                        recency_half_life=recency_half_life,
                        random_state=RANDOM_SEED,
                        learning_rate=gbm_learning_rate,
                        num_boost_round=gbm_rounds,
                        early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                        val_fraction=gbm_val_fraction,
                        log_period=gbm_log_period,
                    )
                probs = _predict_gbm(gbm_model, base_df, feat_cols).astype(float)
                prob_raw = probs.copy()
                p50 = probs
                p05 = np.clip(probs - 0.05, 0.0, 1.0)
                p95 = np.clip(probs + 0.05, 0.0, 1.0)
                if stack_gbm and not use_gbm and logistic_failed:
                    print(
                        "[warn] Bayesian component unavailable due to divergences; "
                        "stacked output falls back to GBM only.",
                        flush=True,
                    )
            elif use_gbm and not stack_gbm:
                if gbm_model is None:
                    gbm_model = _train_gbm_model(
                        train_df, feat_cols,
                        recency_half_life=recency_half_life,
                        random_state=RANDOM_SEED,
                        learning_rate=gbm_learning_rate,
                        num_boost_round=gbm_rounds,
                        early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                        val_fraction=gbm_val_fraction,
                        log_period=gbm_log_period,
                    )
                probs = _predict_gbm(gbm_model, base_df, feat_cols).astype(float)
                prob_raw = probs.copy()
                p50 = probs
                p05 = np.clip(probs - 0.05, 0.0, 1.0)
                p95 = np.clip(probs + 0.05, 0.0, 1.0)
            else:
                probs, p05, p50, p95 = predict_forecast(post_fc, base_df, feat_cols)
                prob_raw = probs.copy()
                if stack_gbm or use_gbm:
                    if gbm_model is None:
                        gbm_model = _train_gbm_model(
                            train_df, feat_cols,
                            recency_half_life=recency_half_life,
                            random_state=RANDOM_SEED,
                            learning_rate=gbm_learning_rate,
                            num_boost_round=gbm_rounds,
                            early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                            val_fraction=gbm_val_fraction,
                            log_period=gbm_log_period,
                        )
                    p_gbm = _predict_gbm(gbm_model, base_df, feat_cols)
                    probs = 0.5 * probs + 0.5 * p_gbm

            if iso is not None and probs.size:
                probs = iso.transform(probs)

            feat_sds = base_df[feat_cols].apply(pd.to_numeric, errors="coerce").std()
            mean_feat_sd = float(feat_sds.mean()) if len(feat_sds) else float("nan")
            mean_prob = float(np.nanmean(probs)) if probs.size else float("nan")
            std_prob = float(np.nanstd(probs)) if probs.size else float("nan")
            print(
                f"[validate] {T:%Y-%m} mean_prob={mean_prob:.3f} std={std_prob:.3f} "
                f"train_pos_rate={train_rate:.3f} mean_feat_sd={mean_feat_sd:.4f}"
            )

            row = {
                "sa2_code": base_df["sa2_code"].values,
                "month": T,
                "price_pressure_prob": probs,
                "prob_p05": p05,
                "prob_p50": p50,
                "prob_p95": p95,
            }
            if iso is not None or (stack_gbm and not use_gbm):
                row["price_pressure_prob_raw"] = prob_raw
            preds.append(pd.DataFrame(row))
        if not preds:
            raise SystemExit("No validation predictions produced (fixed-split).")
        new_df = pd.concat(preds, ignore_index=True)

    else:
        # ===== Walk-forward mode =====
        preds = []
        # We reuse processed SA2 with momentum
        cached_notice_emitted = False
        for T in sorted(val_months):
            base_m = prev_month(T)
            # Define windows:
            # - Nowcast trained on months in [train_start .. base_m]
            # - Logistic trained on months in [train_start .. T-2]
            now_train_months = set(month_range(train_start, base_m))
            logi_train_months = set(month_range(train_start, prev_month(prev_month(T))))
            required_months = now_train_months | {base_m}
            avail_all, missing_cached = _slice_cached_nowcast(cached_nowcast, required_months)
            if avail_all is not None:
                if not cached_notice_emitted:
                    print("[nowcast] Using cached availability for walk-forward validation; skipping PyMC fits.")
                    cached_notice_emitted = True
            else:
                if cached_nowcast is not None and missing_cached:
                    miss = ", ".join(sorted(m.strftime("%Y-%m") for m in sorted(missing_cached)))
                    print(f"[nowcast] Cached availability missing months up to {base_m:%Y-%m}: {miss}. Refitting nowcast…")

                # Fit nowcast for this step
                post_now, mu_mean_train, log_stock_train, train_keys = fit_nowcast_train(
                    sa2, now_train_months, draws=draws, tune=tune, chains=chains, cores=cores,
                    trace_dir=trace_dir,
                    trace_name=f"nowcast_until_{base_m.strftime('%Y-%m')}.nc" if trace_dir else None,
                )
                avail_train = pd.DataFrame({
                    "sa2_code": train_keys["sa2_code"].values,
                    "month": train_keys["month"].values,
                    "availability_rate": (mu_mean_train / np.exp(log_stock_train)),
                })
                # Build availability for base month (predict)
                base_rows = (sa2.loc[sa2["month"] == base_m, ["sa2_code", "month", "stock_bonds"]]
                               .copy().reset_index(drop=True))
                if base_rows.empty:
                    continue
                base_rows["availability_rate"] = predict_availability(post_now, base_rows)
                avail_all = pd.concat([avail_train, base_rows], ignore_index=True)
            if with_spatial:
                avail_all = _add_neighbor_availability(avail_all, adj)

            # Build features and train logistic on [train_start .. T-2]
            df_step = _build_feature_frame(sa2, avail_all, threshold)
            if with_spatial and "nbr_availability_rate" in avail_all.columns:
                df_step = df_step.merge(
                    avail_all[["sa2_code", "month", "nbr_availability_rate"]].drop_duplicates(),
                    on=["sa2_code", "month"], how="left")
                if "nbr_availability_rate" not in feat_cols:
                    feat_cols.append("nbr_availability_rate")
            # === Merge external signals (optional) ===
            if with_external:
                ext_df, ext_cols = _load_external_signals()
                if ext_df is not None and not ext_df.empty:
                    df_step = df_step.merge(ext_df, on="month", how="left")
                    # Create simple lags per SA2 row
                    df_step = df_step.sort_values(["sa2_code","month"])
                    new_cols=[]
                    for c in ext_cols:
                        if c not in feat_cols:
                            feat_cols.append(c)
                        c1=f"{c}_lag1"; c3=f"{c}_lag3"
                        df_step[c1] = df_step.groupby("sa2_code")[c].shift(1)
                        df_step[c3] = df_step.groupby("sa2_code")[c].shift(3)
                        new_cols += [c1,c3]
                    for nc in new_cols:
                        if nc not in feat_cols:
                            feat_cols.append(nc)
            # Extra features for step
            if extra_features:
                df_step = add_calendar_features(df_step)
                for c in ["mo_sin","mo_cos","quarter","is_eofy","is_uni_sem_feb","is_uni_sem_jul"]:
                    if c not in feat_cols:
                        feat_cols.append(c)
                df_step = df_step.sort_values(["sa2_code","month"]).copy()
                df_step["availability_rate_lag1"] = df_step.groupby("sa2_code")["availability_rate"].shift(1)
                df_step["churn_rate_lag1"] = df_step.groupby("sa2_code")["churn_rate"].shift(1)
                df_step["rent_mom_12m"] = (df_step["median_rent"] / df_step.groupby("sa2_code")["median_rent"].shift(12)) - 1.0
                for c in ["availability_rate_lag1","churn_rate_lag1","rent_mom_12m"]:
                    if c not in feat_cols:
                        feat_cols.append(c)

            df_step = add_group_demean(df_step, ["month"], demean_cols, suffix="_wa_dev")
            df_step = add_interaction_features(df_step, interaction_pairs)
            for col in demean_cols:
                name = f"{col}_wa_dev"
                if name not in feat_cols and name in df_step.columns:
                    feat_cols.append(name)
            for a, b in interaction_pairs:
                name = f"{a}__x__{b}"
                if name not in feat_cols and name in df_step.columns:
                    feat_cols.append(name)

            train_df = df_step[df_step["month"].isin(logi_train_months)].copy()
            if train_df.empty:
                continue
            _assert_no_forbidden_features(feat_cols)
            # Predict for T using selected model(s)
            base_df = df_step[df_step["month"] == base_m].copy()
            base_df = base_df.dropna(subset=["availability_rate"])  # ensure availability exists
            if base_df.empty:
                continue

            # Ensure feature list present in base_df
            missing_feats = [c for c in feat_cols if c not in base_df.columns]
            for c in missing_feats:
                base_df[c] = np.nan

            post_fc = None
            logistic_failed = False
            if not (use_gbm and not stack_gbm):
                post_fc, _ = fit_forecast_train(
                    train_df, feat_cols, draws=draws, tune=tune, chains=chains, cores=cores,
                    target_accept=pymc_target_accept,
                    trace_dir=trace_dir,
                    trace_name=f"forecast_until_{prev_month(prev_month(T)).strftime('%Y-%m')}.nc" if trace_dir else None,
                    leakage_canary=leakage_canary,
                    recency_half_life=recency_half_life,
                    rhat_max=sampler_rhat_max,
                    sampler_retries=sampler_retries,
                    retry_draw_multiplier=retry_draw_multiplier,
                )
                if post_fc.divergences:
                    print(
                        f"[warn] PyMC forecast sampler produced {post_fc.divergences} divergences; "
                        "skipping Bayesian logistic component for {T:%Y-%m}.",
                        flush=True,
                    )
                    logistic_failed = True
                    post_fc = None
            gbm_model_step = None
            if use_gbm or stack_gbm:
                gbm_model_step = _train_gbm_model(
                    train_df, feat_cols,
                    recency_half_life=recency_half_life,
                    random_state=RANDOM_SEED,
                    learning_rate=gbm_learning_rate,
                    num_boost_round=gbm_rounds,
                    early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                    val_fraction=gbm_val_fraction,
                    log_period=gbm_log_period,
                )

            # Choose modeling path
            if post_fc is None:
                if gbm_model_step is None:
                    gbm_model_step = _train_gbm_model(
                        train_df, feat_cols,
                        recency_half_life=recency_half_life,
                        random_state=RANDOM_SEED,
                        learning_rate=gbm_learning_rate,
                        num_boost_round=gbm_rounds,
                        early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                        val_fraction=gbm_val_fraction,
                        log_period=gbm_log_period,
                    )
                probs = _predict_gbm(gbm_model_step, base_df, feat_cols).astype(float)
                p05 = np.maximum(0.0, probs - 0.05)
                p50 = probs.copy()
                p95 = np.minimum(1.0, probs + 0.05)
                if stack_gbm and not use_gbm and logistic_failed:
                    print(
                        "[warn] Bayesian component unavailable due to divergences; "
                        f"using GBM-only predictions for {T:%Y-%m}.",
                        flush=True,
                    )
            elif use_gbm and not stack_gbm:
                if gbm_model_step is None:
                    gbm_model_step = _train_gbm_model(
                        train_df, feat_cols,
                        recency_half_life=recency_half_life,
                        random_state=RANDOM_SEED,
                        learning_rate=gbm_learning_rate,
                        num_boost_round=gbm_rounds,
                        early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                        val_fraction=gbm_val_fraction,
                        log_period=gbm_log_period,
                    )
                probs = _predict_gbm(gbm_model_step, base_df, feat_cols).astype(float)
                p05 = np.maximum(0.0, probs - 0.05)
                p50 = probs.copy()
                p95 = np.minimum(1.0, probs + 0.05)
            else:
                # Bayesian logistic (as before)
                probs, p05, p50, p95 = predict_forecast(post_fc, base_df, feat_cols)
                if stack_gbm or use_gbm:
                    # Blend with GBM probability
                    if gbm_model_step is None:
                        gbm_model_step = _train_gbm_model(
                            train_df, feat_cols,
                            recency_half_life=recency_half_life,
                            random_state=RANDOM_SEED,
                            learning_rate=gbm_learning_rate,
                            num_boost_round=gbm_rounds,
                            early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                            val_fraction=gbm_val_fraction,
                            log_period=gbm_log_period,
                        )
                    p_gbm = _predict_gbm(gbm_model_step, base_df, feat_cols)
                    # Simple weight based on train MAE approximations using mean label
                    # Use equal weights if no clean split
                    probs = 0.5 * probs + 0.5 * p_gbm

            # Optional isotonic calibration trained on train_df.
            # If GBM is available, calibrate GBM scores; otherwise calibrate Bayesian probs.
            if calibrate_isotonic:
                try:
                    from sklearn.isotonic import IsotonicRegression  # type: ignore
                    p_train = None
                    if use_gbm or stack_gbm:
                        if gbm_model_step is None:
                            gbm_model_step = _train_gbm_model(
                                train_df, feat_cols,
                                recency_half_life=recency_half_life,
                                random_state=RANDOM_SEED,
                                learning_rate=gbm_learning_rate,
                                num_boost_round=gbm_rounds,
                                early_stopping_rounds=gbm_early_stopping if gbm_early_stopping and gbm_early_stopping > 0 else None,
                                val_fraction=gbm_val_fraction,
                                log_period=gbm_log_period,
                            )
                        p_train = _predict_gbm(gbm_model_step, train_df, feat_cols)
                    if p_train is None and post_fc is not None:
                        # Train a Bayesian model on train_df and predict in-sample for calibration targets
                        _post_tmp, _ = fit_forecast_train(
                            train_df, feat_cols, draws=max(400, draws//2), tune=max(400, tune//2),
                            chains=1, cores=1, target_accept=pymc_target_accept,
                            leakage_canary=leakage_canary,
                            recency_half_life=recency_half_life,
                        )
                        p_train, _, _, _ = predict_forecast(_post_tmp, train_df, feat_cols)
                    y_train = train_df["y"].astype(int).to_numpy()
                    if p_train is not None and len(p_train) and len(set(y_train)) > 1:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(p_train, y_train)
                        probs = iso.transform(probs)
                except Exception:
                    pass
            preds.append(pd.DataFrame({
                "sa2_code": base_df["sa2_code"].values,
                "month": T,
                "price_pressure_prob": probs,
                "prob_p05": p05,
                "prob_p50": p50,
                "prob_p95": p95,
            }))

        if not preds:
            raise SystemExit("No validation predictions produced (walk-forward).")
        new_df = pd.concat(preds, ignore_index=True)

    # Append to history parquet, dedup by (sa2_code, month) keeping newest
    new_df["rent_jump_threshold"] = float(threshold)

    suffix = output_suffix
    if suffix is None and not math.isclose(threshold, RENT_GROWTH_THRESHOLD):
        suffix = _format_threshold_suffix(threshold)
    hist_name = "price_pressure_forecast_sa2_history.parquet" if not suffix else f"price_pressure_forecast_sa2_history_{suffix}.parquet"
    hist_path = STAGE_DIR / hist_name
    try:
        old = pd.read_parquet(hist_path)
        combined = (pd.concat([old, new_df], ignore_index=True)
                    .drop_duplicates(subset=["sa2_code", "month", "rent_jump_threshold"], keep="last")
                    .sort_values(["sa2_code", "month"]))
    except FileNotFoundError:
        combined = new_df.copy()

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(hist_path, index=False)
    months_written = sorted({m.strftime('%Y-%m') for m in new_df['month'].unique()})
    print(
        f"Wrote {len(new_df)} validation predictions @ threshold={threshold:.3f} "
        f"→ {hist_path} (months={months_written})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Time-split train/validate for WA rental forecast")
    ap.add_argument("--train-start", required=True, type=parse_ym)
    ap.add_argument("--train-end", required=True, type=parse_ym)
    ap.add_argument("--val-start", required=True, type=parse_ym)
    ap.add_argument("--val-end", required=True, type=parse_ym)
    ap.add_argument("--threshold", type=float, default=RENT_GROWTH_THRESHOLD,
                    help=f"Rent growth threshold for label (default {RENT_GROWTH_THRESHOLD})")
    ap.add_argument("--threshold-grid", type=float, nargs="+", default=None,
                    help="Optional list of thresholds to iterate sequentially.")
    ap.add_argument("--walk-forward", action="store_true",
                    help="Refit per target month using all data up to T-1 (operational realism)")
    ap.add_argument("--with-external", action="store_true",
                    help="Merge external monthly signals (from data_stage/external_signals.parquet)")
    ap.add_argument("--extra-features", action="store_true",
                    help="Include calendar + WA aggregate features")
    ap.add_argument("--with-spatial", action="store_true",
                    help="Include neighbor-mean availability feature (queen adjacency)")
    ap.add_argument("--leakage-canary", action="store_true",
                    help="Randomly permute train labels to test leakage; accuracy should collapse if no leakage.")
    ap.add_argument("--draws", type=int, default=1000, help="MCMC draws per chain")
    ap.add_argument("--tune", type=int, default=1000, help="MCMC tuning steps per chain")
    ap.add_argument("--chains", type=int, default=4, help="MCMC chains")
    ap.add_argument("--cores", type=int, default=4, help="Worker cores (set 1 for restricted envs)")
    ap.add_argument("--trace-dir", type=str, default=None,
                    help="Directory to save PyMC traces (NetCDF) for verification")
    ap.add_argument("--recency-half-life", type=float, default=FORECAST_RECENCY_HALFLIFE,
                    help="Half-life in months for sample recency weights (default from config).")
    ap.add_argument("--no-recency-weights", action="store_true",
                    help="Disable recency weighting (treat all months equally).")
    ap.add_argument("--use-gbm", action="store_true", help="Use a boosted-tree classifier instead of the Bayesian logistic")
    ap.add_argument("--stack-gbm", action="store_true", help="Blend Bayesian logistic with GBM (simple average)")
    ap.add_argument("--calibrate-isotonic", action="store_true", help="Apply isotonic calibration trained on train window")
    ap.add_argument("--gbm-learning-rate", type=float, default=0.05,
                    help="LightGBM learning rate (default 0.05 for accuracy)" )
    ap.add_argument("--gbm-rounds", type=int, default=400,
                    help="Maximum boosting rounds for LightGBM (default 400)")
    ap.add_argument("--gbm-early-stopping", type=int, default=50,
                    help="Early stopping rounds for LightGBM validation split (set 0 to disable)")
    ap.add_argument("--gbm-val-fraction", type=float, default=0.2,
                    help="Fraction of most-recent months used as LightGBM validation set (default 0.2)")
    ap.add_argument("--gbm-log-period", type=int, default=50,
                    help="Emit LightGBM eval logs every N rounds (set 0 to silence)")
    ap.add_argument("--pymc-target-accept", type=float, default=0.99,
                    help="Target acceptance rate for PyMC NUTS (default 0.99)")
    ap.add_argument("--sampler-rhat-max", type=float, default=1.01,
                    help="Maximum acceptable rank-based R-hat before adding more draws (default 1.01).")
    ap.add_argument("--sampler-retries", type=int, default=2,
                    help="Extra retries with more draws/tune when R-hat is above the limit (default 2).")
    ap.add_argument("--retry-draw-multiplier", type=float, default=1.5,
                    help="Multiplier applied to draws/tune for each retry (default 1.5).")
    ap.add_argument("--force-refit-nowcast", action="store_true",
                    help="Ignore cached availability and refit the PyMC nowcast")
    ap.add_argument("--output-suffix", type=str, default=None,
                    help="Optional suffix for output files (overrides auto suffix when using alternate thresholds).")
    args = ap.parse_args()

    from pathlib import Path as _Path
    tdir = _Path(args.trace_dir) if args.trace_dir else None
    thresholds = args.threshold_grid if args.threshold_grid else [args.threshold]

    for thr in thresholds:
        suffix = args.output_suffix
        if suffix is None and not math.isclose(thr, RENT_GROWTH_THRESHOLD):
            suffix = _format_threshold_suffix(thr)
        elif suffix is not None and len(thresholds) > 1:
            # Ensure unique suffix per threshold when user supplies base suffix
            suffix = f"{suffix}_{_format_threshold_suffix(thr)}"

        main(args.train_start, args.train_end, args.val_start, args.val_end,
             threshold=thr, walk_forward=args.walk_forward,
             draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
             trace_dir=tdir, leakage_canary=args.leakage_canary, with_external=args.with_external,
             extra_features=args.extra_features, recency_half_life=None if args.no_recency_weights else args.recency_half_life,
             use_gbm=args.use_gbm, stack_gbm=args.stack_gbm, calibrate_isotonic=args.calibrate_isotonic,
             gbm_learning_rate=args.gbm_learning_rate, gbm_rounds=args.gbm_rounds,
             gbm_early_stopping=args.gbm_early_stopping,
             gbm_val_fraction=args.gbm_val_fraction, gbm_log_period=args.gbm_log_period,
             pymc_target_accept=args.pymc_target_accept,
             sampler_rhat_max=args.sampler_rhat_max,
             sampler_retries=args.sampler_retries,
             retry_draw_multiplier=args.retry_draw_multiplier,
             force_refit_nowcast=args.force_refit_nowcast,
             output_suffix=suffix)
