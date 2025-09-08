"""
Time-split training and validation for the WA rental forecast.

Implements a strict split:
 - Fit the availability nowcast (Negative Binomial with offset) on a train window.
 - Use that nowcast to produce out-of-sample availability_rate for base months.
 - Fit the hierarchical logistic forecast on the train window using those features.
 - Predict probabilities for a validation window (months val_start..val_end),
   using base-month features (month T-1).

Writes predictions into data_stage/price_pressure_forecast_sa2_history.parquet
so that src.evaluate_forecasts can score them (and the static site can consume).

Usage:
  python -m src.time_split_validate \
    --train-start 2025-01 --train-end 2025-04 \
    --val-start 2025-06 --val-end 2025-09
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import pymc as pm

from src.config import STAGE_DIR, RENT_GROWTH_THRESHOLD, RANDOM_SEED


# ----------------- small helpers -----------------

def to_month(ts_like) -> pd.Series | pd.Timestamp:
    dt = pd.to_datetime(ts_like)
    # If Series/Index-like, use .dt accessor
    if hasattr(dt, "dt"):
        return dt.dt.to_period("M").dt.to_timestamp()
    return dt.to_period("M").to_timestamp()


def parse_ym(s: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(s).to_period("M").to_timestamp()
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid YYYY-MM: {s}") from e


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if end < start:
        return []
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    return [p.to_timestamp() for p in months]


def prev_month(m: pd.Timestamp) -> pd.Timestamp:
    return (m.to_period("M") - 1).to_timestamp()


def ensure_cols(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in SA2 panel: {missing}")


def add_rent_momentum(sa2: pd.DataFrame) -> pd.DataFrame:
    df = sa2.sort_values(["sa2_code", "month"]).copy()
    df["rent_mom_1m"] = df.groupby("sa2_code")["median_rent"].pct_change(1, fill_method=None)
    df["rent_mom_3m"] = (df["median_rent"] / df.groupby("sa2_code")["median_rent"].shift(3)) ** (1/3) - 1
    return df


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


def _build_month_index(all_months: Iterable[pd.Timestamp]) -> dict:
    uniq = sorted({to_month(m) for m in all_months})
    return {m: i for i, m in enumerate(uniq)}


def fit_nowcast_train(df: pd.DataFrame,
                      train_months: set[pd.Timestamp],
                      *, draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4,
                      target_accept: float = 0.95,
                      trace_dir: 'Path | None' = None,
                      trace_name: str | None = None) -> Tuple[NowcastPosterior, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Fit the NB nowcast on train_months only. Returns:
      - NowcastPosterior (stacked draws of components)
      - mu_mean_train (per-observation posterior mean mu for train rows)
      - log_stock_train
      - sa2_cat_train (Categorical codes)
    """
    # Select train rows and build indices
    dft = df[df["month"].isin(train_months)].copy()
    dft = dft.dropna(subset=["stock_bonds"]).sort_values(["sa2_code", "month"])  # consistent with src/model_nowcast

    ensure_cols(dft, ["count_disposals", "stock_bonds"])  # response/exposure

    y = dft["count_disposals"].fillna(0).astype(int).to_numpy()
    log_stock = np.log(dft["stock_bonds"].clip(lower=1.0).to_numpy())

    # Month indices over the full timeline (train+val) for consistent t
    months_index = _build_month_index(df["month"].unique())
    t = dft["month"].map(months_index).astype(int).to_numpy()
    season = dft["month"].dt.month.astype(int).to_numpy() - 1

    sa2_codes = pd.Index(sorted(dft["sa2_code"].unique()))
    sa2_cat = pd.Categorical(dft["sa2_code"], categories=sa2_codes)
    sa2_idx = sa2_cat.codes.astype(int)
    n_sa2 = len(sa2_codes)

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

        pm.NegativeBinomial("y", mu=mu, alpha=alpha_nb, observed=y)

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
        months_index=months_index,
    )
    # Return keys for alignment (same order as y/log_stock/mu_mean)
    keys = dft[["sa2_code", "month"]].reset_index(drop=True)
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


def fit_forecast_train(train_df: pd.DataFrame,
                       feat_cols: list[str], *, draws: int = 1000, tune: int = 1000,
                       chains: int = 4, cores: int = 4, target_accept: float = 0.9,
                       trace_dir: 'Path | None' = None,
                       trace_name: str | None = None) -> Tuple[ForecastPosterior, pd.DataFrame]:
    # Prepare features/labels (impute NaNs with feature-wise train means to keep early months)
    df = train_df.dropna(subset=["y"]).copy()
    X = df[feat_cols].to_numpy()
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    X = np.where(np.isnan(X), mu, X)
    Xz = (X - mu) / sd

    sa2_codes = pd.Index(sorted(df["sa2_code"].unique()))
    sa2_cat = pd.Categorical(df["sa2_code"], categories=sa2_codes)
    sa2_idx = sa2_cat.codes.astype(int)
    y = df["y"].astype(int).to_numpy()
    n_sa2 = len(sa2_codes)
    n, k = Xz.shape

    with pm.Model() as m:
        beta = pm.Normal("beta", 0.0, 1.0, shape=k)
        mu_a = pm.Normal("mu_a", 0.0, 1.0)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)
        eta = a_sa2[sa2_idx] + pm.math.dot(Xz, beta)
        pm.Bernoulli("y", logit_p=eta, observed=y)
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

    post = ForecastPosterior(
        beta=beta_s,
        a_sa2=a_sa2_s,
        sa2_index={code: i for i, code in enumerate(sa2_codes)},
        feat_mu=mu,
        feat_sd=sd,
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
    X = base_df[feat_cols].to_numpy()
    X = np.where(np.isnan(X), post.feat_mu, X)
    Xz = (X - post.feat_mu) / post.feat_sd

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
        ps = 1.0 / (1.0 + np.exp(-lin))
        probs[i] = ps.mean()
        p05[i] = np.quantile(ps, 0.05)
        p50[i] = np.quantile(ps, 0.50)
        p95[i] = np.quantile(ps, 0.95)
    return probs, p05, p50, p95


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
    df["churn_rate"] = df["count_disposals"] / df["stock_bonds"].replace({0: np.nan})
    return df


def main(train_start: pd.Timestamp,
         train_end: pd.Timestamp,
         val_start: pd.Timestamp,
         val_end: pd.Timestamp,
         threshold: float = RENT_GROWTH_THRESHOLD,
         walk_forward: bool = False,
         draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4,
         trace_dir: 'Path | None' = None):
    # Load SA2 panel
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    ensure_cols(sa2, ["sa2_code", "month", "median_rent", "count_disposals", "stock_bonds"])
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

    new_df = None

    if not walk_forward:
        # ===== Fixed-split mode =====
        # Fit nowcast on train window only
        post_now, mu_mean_train, log_stock_train, train_keys = fit_nowcast_train(
            sa2, train_months, draws=draws, tune=tune, chains=chains, cores=cores,
            trace_dir=trace_dir,
            trace_name=f"nowcast_{train_start.strftime('%Y-%m')}_{train_end.strftime('%Y-%m')}.nc" if trace_dir else None,
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

        df = _build_feature_frame(sa2, avail_all, threshold)

        # Train logistic on train window
        train_df = df[df["month"].isin(train_months)].copy()
        post_fc, _ = fit_forecast_train(
            train_df, feat_cols, draws=draws, tune=tune, chains=chains, cores=cores,
            trace_dir=trace_dir,
            trace_name=(f"forecast_{train_start.strftime('%Y-%m')}_{train_end.strftime('%Y-%m')}"
                        f"_val_{val_start.strftime('%Y-%m')}_{val_end.strftime('%Y-%m')}.nc") if trace_dir else None,
        )

        # Predict target months
        preds = []
        for T in sorted(val_months):
            base_m = prev_month(T)
            base_df = df[df["month"] == base_m].copy()
            if base_df.empty:
                continue
            base_df = base_df.dropna(subset=["availability_rate"])  # ensure availability exists
            probs, p05, p50, p95 = predict_forecast(post_fc, base_df, feat_cols)
            preds.append(pd.DataFrame({
                "sa2_code": base_df["sa2_code"].values,
                "month": T,
                "price_pressure_prob": probs,
                "prob_p05": p05,
                "prob_p50": p50,
                "prob_p95": p95,
            }))
        if not preds:
            raise SystemExit("No validation predictions produced (fixed-split).")
        new_df = pd.concat(preds, ignore_index=True)

    else:
        # ===== Walk-forward mode =====
        preds = []
        # We reuse processed SA2 with momentum
        for T in sorted(val_months):
            base_m = prev_month(T)
            # Define windows:
            # - Nowcast trained on months in [train_start .. base_m]
            # - Logistic trained on months in [train_start .. T-2]
            now_train_months = set(month_range(train_start, base_m))
            logi_train_months = set(month_range(train_start, prev_month(prev_month(T))))

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

            # Build features and train logistic on [train_start .. T-2]
            df_step = _build_feature_frame(sa2, avail_all, threshold)
            train_df = df_step[df_step["month"].isin(logi_train_months)].copy()
            if train_df.empty:
                continue
            post_fc, _ = fit_forecast_train(
                train_df, feat_cols, draws=draws, tune=tune, chains=chains, cores=cores,
                trace_dir=trace_dir,
                trace_name=f"forecast_until_{prev_month(prev_month(T)).strftime('%Y-%m')}.nc" if trace_dir else None,
            )

            # Predict for T using base month row
            base_df = df_step[df_step["month"] == base_m].copy()
            base_df = base_df.dropna(subset=["availability_rate"])  # ensure availability exists
            if base_df.empty:
                continue
            probs, p05, p50, p95 = predict_forecast(post_fc, base_df, feat_cols)
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
    ap.add_argument("--draws", type=int, default=1000, help="MCMC draws per chain")
    ap.add_argument("--tune", type=int, default=1000, help="MCMC tuning steps per chain")
    ap.add_argument("--chains", type=int, default=4, help="MCMC chains")
    ap.add_argument("--cores", type=int, default=4, help="Worker cores (set 1 for restricted envs)")
    ap.add_argument("--trace-dir", type=str, default=None,
                    help="Directory to save PyMC traces (NetCDF) for verification")
    args = ap.parse_args()

    from pathlib import Path as _Path
    tdir = _Path(args.trace_dir) if args.trace_dir else None
    main(args.train_start, args.train_end, args.val_start, args.val_end,
         threshold=args.threshold, walk_forward=args.walk_forward,
         draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
         trace_dir=tdir)
