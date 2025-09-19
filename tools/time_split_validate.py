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
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import pymc as pm

from src.config import STAGE_DIR, RENT_GROWTH_THRESHOLD, RANDOM_SEED
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
    """Return month-level WA aggregates computed from SA2 panel.
    Columns: month, wa_rent_mom, wa_churn_rate, wa_stock_growth, wa_disp_rate
    """
    df = sa2.copy()
    df = df.sort_values(['sa2_code','month'])
    # Compute prev stock per SA2 for growth later
    df['stock_prev'] = df.groupby('sa2_code')['stock_bonds'].shift(1)
    # Weighted mean rent momentum by stock (per month)
    rm = (
        df.assign(w=lambda x: x['stock_bonds'].fillna(0.0),
                  wm=lambda x: x['rent_mom_1m'].fillna(0.0) * x['stock_bonds'].fillna(0.0))
          .groupby('month')
          .apply(lambda x: x['wm'].sum() / max(x['w'].sum(), 1.0))
          .to_frame('wa_rent_mom')
          .reset_index()
    )
    # Churn and stock growth at WA level
    tot = (df.groupby('month', as_index=False)
             .agg(count_lodgements=('count_lodgements','sum'),
                  count_disposals=('count_disposals','sum'),
                  stock_bonds=('stock_bonds','sum'),
                  stock_prev=('stock_prev','sum')))
    tot['wa_churn_rate'] = (tot['count_lodgements'] + tot['count_disposals']) / tot['stock_bonds'].clip(lower=1.0)
    tot['wa_disp_rate'] = (tot['count_disposals']) / tot['stock_bonds'].clip(lower=1.0)
    with pd.option_context('mode.use_inf_as_na', True):
        tot['wa_stock_growth'] = tot['stock_bonds'] / tot['stock_prev'] - 1.0
    tot['wa_stock_growth'] = tot['wa_stock_growth'].fillna(0.0)
    out = tot[['month','wa_churn_rate','wa_disp_rate','wa_stock_growth']].merge(rm, on='month', how='left')
    return out.sort_values('month').reset_index(drop=True)
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
    beta: np.ndarray     # (S, k)
    a_sa2: np.ndarray    # (S, n_sa2_train)
    sa2_index: dict      # code -> idx
    feat_mu: np.ndarray  # (k,) training standardization mean
    feat_sd: np.ndarray  # (k,) training standardization std
    divergences: int = 0


def fit_forecast_train(train_df: pd.DataFrame,
                       feat_cols: list[str], *, draws: int = 1000, tune: int = 1000,
                       chains: int = 4, cores: int = 4, target_accept: float = 0.99,
                       trace_dir: 'Path | None' = None,
                       trace_name: str | None = None,
                       leakage_canary: bool = False,
                       recency_half_life: float | None = None) -> Tuple[ForecastPosterior, pd.DataFrame]:
    # Prepare features/labels (impute NaNs with feature-wise train means to keep early months)
    df = train_df.dropna(subset=["y"]).copy()
    Xz, mu, sd = standardize_columns(df, feat_cols)
    Xz = np.clip(Xz, -5.0, 5.0)

    sa2_codes = pd.Index(sorted(df["sa2_code"].unique()))
    sa2_cat = pd.Categorical(df["sa2_code"], categories=sa2_codes)
    sa2_idx = sa2_cat.codes.astype(int)
    y = df["y"].astype(int).to_numpy()
    if leakage_canary:
        # Randomly permute labels to collapse any leakage-driven signal
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(y)

    # Optional recency weights (half-life in months; newer rows get higher weight)
    w = compute_recency_weights(df["month"], recency_half_life)

    n_sa2 = len(sa2_codes)
    n, k = Xz.shape

    with pm.Model() as m:
        beta = pm.Normal("beta", 0.0, 0.5, shape=k)
        mu_a = pm.Normal("mu_a", 0.0, 0.5)
        sigma_a = pm.HalfNormal("sigma_a", 0.3)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)
        eta = a_sa2[sa2_idx] + pm.math.dot(Xz, beta)
        if w is None:
            pm.Bernoulli("y", logit_p=eta, observed=y)
        else:
            y_dist = pm.Bernoulli.dist(logit_p=eta)
            pm.Potential("weighted_loglik_binom", (w * pm.logp(y_dist, y)).sum())
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept,
            random_seed=RANDOM_SEED, progressbar=True
        )
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

    divergences = int(idata.sample_stats["diverging"].sum().item())

    post = ForecastPosterior(
        beta=beta_s,
        a_sa2=a_sa2_s,
        sa2_index={code: i for i, code in enumerate(sa2_codes)},
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
    Xz = np.clip(Xz, -5.0, 5.0)

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
         recency_half_life: float | None = None,
         use_gbm: bool = False,
         stack_gbm: bool = False,
         calibrate_isotonic: bool = False,
         gbm_learning_rate: float = 0.05,
         gbm_rounds: int = 400,
         gbm_early_stopping: int | None = 50,
         gbm_val_fraction: float = 0.2,
         gbm_log_period: int | None = 50,
         pymc_target_accept: float = 0.99):
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
    cached_nowcast = _load_cached_nowcast()

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
    hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
    try:
        old = pd.read_parquet(hist_path)
        combined = (pd.concat([old, new_df], ignore_index=True)
                    .drop_duplicates(subset=["sa2_code", "month"], keep="last")
                    .sort_values(["sa2_code", "month"]))
    except FileNotFoundError:
        combined = new_df.copy()

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(hist_path, index=False)
    print(f"Wrote {len(new_df)} validation predictions → {hist_path} (months={sorted({m.strftime('%Y-%m') for m in new_df['month'].unique()})})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Time-split train/validate for WA rental forecast")
    ap.add_argument("--train-start", required=True, type=parse_ym)
    ap.add_argument("--train-end", required=True, type=parse_ym)
    ap.add_argument("--val-start", required=True, type=parse_ym)
    ap.add_argument("--val-end", required=True, type=parse_ym)
    ap.add_argument("--threshold", type=float, default=RENT_GROWTH_THRESHOLD,
                    help=f"Rent growth threshold for label (default {RENT_GROWTH_THRESHOLD})")
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
    ap.add_argument("--recency-half-life", type=float, default=None,
                    help="Half-life in months for sample recency weights (e.g., 12 → 50% weight per year older)")
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
    args = ap.parse_args()

    from pathlib import Path as _Path
    tdir = _Path(args.trace_dir) if args.trace_dir else None
    main(args.train_start, args.train_end, args.val_start, args.val_end,
         threshold=args.threshold, walk_forward=args.walk_forward,
         draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
         trace_dir=tdir, leakage_canary=args.leakage_canary, with_external=args.with_external,
         extra_features=args.extra_features, recency_half_life=args.recency_half_life,
         use_gbm=args.use_gbm, stack_gbm=args.stack_gbm, calibrate_isotonic=args.calibrate_isotonic,
         gbm_learning_rate=args.gbm_learning_rate, gbm_rounds=args.gbm_rounds,
         gbm_early_stopping=args.gbm_early_stopping,
         gbm_val_fraction=args.gbm_val_fraction, gbm_log_period=args.gbm_log_period,
         pymc_target_accept=args.pymc_target_accept)
