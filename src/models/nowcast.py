"""
Negative Binomial nowcast of disposals with stock as exposure.

Builds features from the SA2 panel and fits an NB model with:
 - SA2 random intercepts
 - Linear time trend
 - Month-of-year seasonality (sum-to-zero)

Writes `data_stage/availability_nowcast_sa2.parquet` with availability_rate.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import argparse

from src.config import STAGE_DIR, RANDOM_SEED
from src.features.dates import compute_recency_weights
from src.models.nowcast_design import prepare_design

def build_design():
    """Load SA2 panel and return standardized nowcast design arrays."""
    df = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    return prepare_design(df)

def fit_nowcast(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    recency_half_life: float | None = None,
    target_accept: float = 0.98,
    max_treedepth: int = 15,
):
    """Fit the NB nowcast and write availability rate parquet.

    Returns the InferenceData object for optional diagnostics.
    """
    design = build_design()
    df = design.frame
    y = design.y
    log_stock = design.log_stock
    t = design.time_index
    season = design.season_index
    sa2_idx = design.sa2_index
    n_sa2 = design.n_sa2
    N = design.n_obs

    # Optional recency weights (half-life in months; newer rows ↑ weight)
    w = compute_recency_weights(df["month"], recency_half_life)

    with pm.Model() as m:
        # SA2 random intercept
        mu_a = pm.Normal("mu_a", 0.0, 1.0)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)

        # Linear time trend
        beta_t = pm.Normal("beta_t", 0.0, 0.2)

        # Month-of-year (sum-to-zero)
        sigma_season = pm.HalfNormal("sigma_season", 0.5)
        season_raw = pm.Normal("season_raw", 0.0, 1.0, shape=12)
        season_eff = pm.Deterministic("season_eff", season_raw - pm.math.mean(season_raw))

        # Overdispersion (slightly tighter prior to improve geometry)
        alpha_nb = pm.HalfNormal("alpha_nb", 1.0)

        # Linear predictor with offset
        eta = a_sa2[sa2_idx] + beta_t * t + season_eff[season] + log_stock
        mu = pm.Deterministic("mu", pm.math.exp(eta))

        if w is None:
            pm.NegativeBinomial("y", mu=mu, alpha=alpha_nb, observed=y)
        else:
            y_dist = pm.NegativeBinomial.dist(mu=mu, alpha=alpha_nb)
            pm.Potential("weighted_loglik_nb", (w * pm.logp(y_dist, y)).sum())

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            step=pm.NUTS(max_treedepth=max_treedepth),
            random_seed=RANDOM_SEED,
            progressbar=True,
        )

    # Posterior mean of mu per observation (chain, draw, obs) → (obs,)
    mu_mean = idata.posterior["mu"].mean(dim=("chain", "draw")).values.squeeze()
    # Availability rate = expected counts per unit stock
    availability_rate = mu_mean / np.exp(log_stock)

    out = df[["sa2_code", "month"]].copy()
    out["availability_rate"] = availability_rate
    out.to_parquet(STAGE_DIR / "availability_nowcast_sa2.parquet", index=False)
    print("Wrote availability nowcast → data_stage/availability_nowcast_sa2.parquet")
    return idata

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit NB nowcast with offset and write availability file")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--recency-half-life", type=float, default=None,
                    help="Half-life in months for sample recency weights (e.g., 12 → 50% per year older)")
    ap.add_argument("--target-accept", type=float, default=0.98,
                    help="Target acceptance rate for NUTS (default 0.98)")
    ap.add_argument("--max-treedepth", type=int, default=15,
                    help="Maximum tree depth for NUTS (default 15)")
    args = ap.parse_args()
    fit_nowcast(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        recency_half_life=args.recency_half_life,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
    )
