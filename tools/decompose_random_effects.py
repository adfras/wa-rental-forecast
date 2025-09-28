"""
Decompose the hierarchical logistic forecast into fixed vs random effects.

Computes:
 - Variance share of the linear predictor explained by:
     * Fixed effects (X @ beta)
     * Random intercepts (a_sa2)
     * Random varying slopes (if enabled)
 - Nakagawa & Schielzeth style R^2_marginal and R^2_conditional for GLMMs
   (logistic approximation uses residual variance = pi^2/3).
 - ICC for the random intercept: Var(a) / (Var(a) + pi^2/3)

Usage:
  python -m tools.decompose_random_effects --draws 600 --tune 600 --chains 2 --cores 2 \
      --target-accept 0.995
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import pymc as pm

from src.models.forecast import build_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decompose forecast into fixed vs random effects")
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--tune", type=int, default=600)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--target-accept", type=float, default=0.995)
    ap.add_argument("--max-treedepth", type=int, default=15)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df_model, Xz, y, sa2_idx, sa2_codes, feat_cols, mu, sd, _ = build_dataset()
    n_sa2 = len(sa2_codes)
    n, k = Xz.shape

    # Indices for key features (varying slopes)
    idx_avail = feat_cols.index("availability_rate") if "availability_rate" in feat_cols else None
    idx_churn = feat_cols.index("churn_rate") if "churn_rate" in feat_cols else None
    idx_price = feat_cols.index("price_band_high") if "price_band_high" in feat_cols else None
    idx_rm1 = feat_cols.index("rent_mom_1m") if "rent_mom_1m" in feat_cols else None

    pi = float(y.mean()) if len(y) else 0.5
    pi = max(1e-4, min(1 - 1e-4, pi))
    mu_intercept = np.log(pi / (1.0 - pi))
    enable_varying = os.getenv("VARYING_SLOPES", "1").lower() in {"1", "true", "yes"}

    with pm.Model() as m:
        beta = pm.Normal("beta", 0.0, 0.3, shape=k)
        mu_a = pm.Normal("mu_a", mu_intercept, 0.5)
        sigma_a = pm.HalfNormal("sigma_a", 0.5)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)

        eta = a_sa2[sa2_idx] + pm.math.dot(Xz, beta)
        b_av_sa2 = None
        b_ch_sa2 = None
        b_pr_sa2 = None
        b_rm1_sa2 = None
        if enable_varying and idx_avail is not None:
            mu_b_av = pm.Normal("mu_b_av", 0.0, 0.2)
            sigma_b_av = pm.HalfNormal("sigma_b_av", 0.3)
            z_b_av = pm.Normal("z_b_av", 0.0, 1.0, shape=n_sa2)
            b_av_sa2 = pm.Deterministic("b_av_sa2", mu_b_av + z_b_av * sigma_b_av)
            eta = eta + b_av_sa2[sa2_idx] * Xz[:, idx_avail]
        if enable_varying and idx_churn is not None:
            mu_b_ch = pm.Normal("mu_b_ch", 0.0, 0.2)
            sigma_b_ch = pm.HalfNormal("sigma_b_ch", 0.3)
            z_b_ch = pm.Normal("z_b_ch", 0.0, 1.0, shape=n_sa2)
            b_ch_sa2 = pm.Deterministic("b_ch_sa2", mu_b_ch + z_b_ch * sigma_b_ch)
            eta = eta + b_ch_sa2[sa2_idx] * Xz[:, idx_churn]
        if enable_varying and idx_price is not None:
            mu_b_pr = pm.Normal("mu_b_pr", 0.0, 0.2)
            sigma_b_pr = pm.HalfNormal("sigma_b_pr", 0.3)
            z_b_pr = pm.Normal("z_b_pr", 0.0, 1.0, shape=n_sa2)
            b_pr_sa2 = pm.Deterministic("b_pr_sa2", mu_b_pr + z_b_pr * sigma_b_pr)
            eta = eta + b_pr_sa2[sa2_idx] * Xz[:, idx_price]
        if enable_varying and idx_rm1 is not None:
            mu_b_rm1 = pm.Normal("mu_b_rm1", 0.0, 0.2)
            sigma_b_rm1 = pm.HalfNormal("sigma_b_rm1", 0.3)
            z_b_rm1 = pm.Normal("z_b_rm1", 0.0, 1.0, shape=n_sa2)
            b_rm1_sa2 = pm.Deterministic("b_rm1_sa2", mu_b_rm1 + z_b_rm1 * sigma_b_rm1)
            eta = eta + b_rm1_sa2[sa2_idx] * Xz[:, idx_rm1]

        pm.Bernoulli("y", logit_p=eta, observed=y)

        try:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                step=pm.NUTS(target_accept=args.target_accept, max_treedepth=args.max_treedepth),
                progressbar=True,
            )
            mode = "mcmc"
        except Exception as e:
            # Fallback to MAP for a fast, approximate decomposition
            print(f"[warn] MCMC sampling failed ({e}). Falling back to MAP approximation.")
            mp = pm.find_MAP()
            class _FakeID:
                posterior: dict
            idata = _FakeID()
            idata.posterior = {
                "beta": np.expand_dims(mp["beta"], (0,1)),
                "a_sa2": np.expand_dims(mp["a_sa2"], (0,1)),
                **({"b_av_sa2": np.expand_dims(mp["b_av_sa2"], (0,1))} if "b_av_sa2" in mp else {}),
                **({"b_ch_sa2": np.expand_dims(mp["b_ch_sa2"], (0,1))} if "b_ch_sa2" in mp else {}),
                **({"b_pr_sa2": np.expand_dims(mp["b_pr_sa2"], (0,1))} if "b_pr_sa2" in mp else {}),
                **({"b_rm1_sa2": np.expand_dims(mp["b_rm1_sa2"], (0,1))} if "b_rm1_sa2" in mp else {}),
                "sigma_a": np.array([[mp.get("sigma_a", np.nan)]]),
            }
            mode = "map"

    # Posterior means for components
    def mean_of(name):
        da = idata.posterior[name]
        # support both InferenceData (xarray) and numpy dict from MAP fallback
        try:
            import xarray as xr  # type: ignore
            if isinstance(da, xr.DataArray):
                return np.asarray(da.mean(dim=("chain", "draw")).values)
        except Exception:
            pass
        arr = np.asarray(da)
        while arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr

    beta_mean = mean_of("beta")  # (k,)
    a_sa2_mean = mean_of("a_sa2")  # (n_sa2,)
    FE = Xz @ beta_mean
    # Per-feature contribution variances (using posterior-mean beta)
    per_feat_var = []
    for j, name in enumerate(feat_cols):
        vj = float(np.var(Xz[:, j] * float(beta_mean[j])))
        per_feat_var.append((name, vj))
    per_feat_var.sort(key=lambda t: t[1], reverse=True)
    REi = a_sa2_mean[sa2_idx]
    REV = np.zeros_like(REi)
    if "b_av_sa2" in idata.posterior:
        b_av_mean = mean_of("b_av_sa2")  # (n_sa2,)
        REV = REV + b_av_mean[sa2_idx] * Xz[:, idx_avail]
    if "b_ch_sa2" in idata.posterior:
        b_ch_mean = mean_of("b_ch_sa2")
        REV = REV + b_ch_mean[sa2_idx] * Xz[:, idx_churn]
    if "b_pr_sa2" in idata.posterior and idx_price is not None:
        b_pr_mean = mean_of("b_pr_sa2")
        REV = REV + b_pr_mean[sa2_idx] * Xz[:, idx_price]
    if "b_rm1_sa2" in idata.posterior and idx_rm1 is not None:
        b_rm1_mean = mean_of("b_rm1_sa2")
        REV = REV + b_rm1_mean[sa2_idx] * Xz[:, idx_rm1]

    eta_mean = FE + REi + REV
    v_total = float(np.var(eta_mean))
    v_fe = float(np.var(FE))
    v_rei = float(np.var(REi))
    v_rev = float(np.var(REV))
    # Correlations matter; shares won't sum to 1 unless components are orthogonal.
    # Report both raw variance shares and Nakagawa R^2 decomposition.

    # Nakagawa R^2 (approx) with logistic residual variance = pi^2/3
    v_resid = np.pi**2 / 3.0
    v_random = v_rei + v_rev
    r2_marg = v_fe / (v_fe + v_random + v_resid) if (v_fe + v_random + v_resid) > 0 else np.nan
    r2_cond = (v_fe + v_random) / (v_fe + v_random + v_resid) if (v_fe + v_random + v_resid) > 0 else np.nan

    # ICC for random intercept
    try:
        import xarray as xr  # type: ignore
        if isinstance(idata.posterior.get("sigma_a"), xr.DataArray):
            sigma_a = float(idata.posterior["sigma_a"].mean().values)
        else:
            # MAP fallback may store transformed var names
            if "sigma_a" in idata.posterior:
                sigma_a = float(np.asarray(idata.posterior["sigma_a"]).mean())
            elif "sigma_a_log__" in idata.posterior:
                sigma_a = float(np.exp(np.asarray(idata.posterior["sigma_a_log__"]).mean()))
            else:
                sigma_a = float("nan")
        icc = sigma_a**2 / (sigma_a**2 + v_resid) if np.isfinite(sigma_a) else float("nan")
    except Exception:
        sigma_a = float("nan"); icc = float("nan")

    print(f"--- Linear predictor variance (logit scale) [{mode}] ---")
    print(f"Var(total eta): {v_total:.4f}")
    print(f"Var(FE):        {v_fe:.4f}  | share vs total: {v_fe/max(v_total,1e-12):.2%}")
    print(f"Var(RE intercepts): {v_rei:.4f}  | share vs total: {v_rei/max(v_total,1e-12):.2%}")
    print(f"Var(RE varying):    {v_rev:.4f}  | share vs total: {v_rev/max(v_total,1e-12):.2%}")
    print("Note: shares need not sum to 100% due to covariance among components.")
    print()
    print("--- GLMM R^2 (Nakagawa, logistic approx) ---")
    print(f"R2_marginal (fixed only):   {r2_marg:.3f}")
    print(f"R2_conditional (fixed+REs): {r2_cond:.3f}")
    print(f"ICC (random intercept, mean): {icc:.3f}  (sigma_a~{sigma_a:.3f})")
    print()
    print("Top fixed-effect drivers (by Var(beta_j * x_j), logit-scale):")
    for name, vj in per_feat_var[:10]:
        print(f"  {name:30s}  Var contrib: {vj:.4f}")


if __name__ == "__main__":
    main()
