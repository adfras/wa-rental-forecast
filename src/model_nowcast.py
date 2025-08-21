import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from src.config import STAGE_DIR, RANDOM_SEED

def build_data():
    df = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    df = df.dropna(subset=["stock_bonds"]).sort_values(["sa2_code", "month"]).reset_index(drop=True)

    # Response = disposals (you've been using this in the SA2 panel)
    y = df["count_disposals"].fillna(0).astype(int).to_numpy()

    # Offset (exposure) = log(stock)
    log_stock = np.log(df["stock_bonds"].clip(lower=1.0).to_numpy())

    # Time index and month-of-year
    t = ((df["month"] - df["month"].min()).dt.days // 30).astype(int).to_numpy()
    season = df["month"].dt.month.astype(int).to_numpy() - 1  # 0..11

    # SA2 index
    sa2_codes = pd.Index(sorted(df["sa2_code"].unique()))
    sa2_idx = pd.Categorical(df["sa2_code"], categories=sa2_codes).codes.astype(int)

    return df, y, log_stock, t, season, sa2_idx, len(sa2_codes)

def fit_nowcast():
    df, y, log_stock, t, season, sa2_idx, n_sa2 = build_data()
    N = len(y)

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

        # Overdispersion
        alpha_nb = pm.HalfNormal("alpha_nb", 1.5)

        # Linear predictor with offset
        eta = a_sa2[sa2_idx] + beta_t * t + season_eff[season] + log_stock
        mu = pm.Deterministic("mu", pm.math.exp(eta))

        pm.NegativeBinomial("y", mu=mu, alpha=alpha_nb, observed=y)

        idata = pm.sample(
            draws=1000, tune=1000, chains=4, target_accept=0.95,
            random_seed=RANDOM_SEED, progressbar=True
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
    fit_nowcast()
