import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from src.config import STAGE_DIR, RENT_GROWTH_THRESHOLD, RANDOM_SEED

def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))

def build_dataset():
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    now = pd.read_parquet(STAGE_DIR / "availability_nowcast_sa2.parquet").copy()

    # Align month types
    sa2["month"] = pd.to_datetime(sa2["month"]).dt.to_period("M").dt.to_timestamp()
    now["month"] = pd.to_datetime(now["month"]).dt.to_period("M").dt.to_timestamp()

    df = sa2.merge(now, on=["sa2_code", "month"], how="left")

    # Momentum features (silence FutureWarning via fill_method=None)
    df = df.sort_values(["sa2_code", "month"])
    df["rent_mom_1m"] = df.groupby("sa2_code")["median_rent"].pct_change(1, fill_method=None)
    df["rent_mom_3m"] = (df["median_rent"] / df.groupby("sa2_code")["median_rent"].shift(3)) ** (1/3) - 1

    # Churn proxy
    if {"count_disposals", "stock_bonds"} <= set(df.columns):
        df["churn_rate"] = df["count_disposals"] / df["stock_bonds"].replace({0: np.nan})
    elif "mean_days_held" in df.columns:
        df["churn_rate"] = 30.0 / df["mean_days_held"]
    else:
        df["churn_rate"] = np.nan

    # Label: next-month rent rise > threshold
    df["median_rent_next"] = df.groupby("sa2_code")["median_rent"].shift(-1)
    df["y"] = ((df["median_rent_next"] - df["median_rent"]) / df["median_rent"] > RENT_GROWTH_THRESHOLD).astype(float)

    # Features
    feat_cols = ["availability_rate", "churn_rate", "rent_mom_1m", "rent_mom_3m"]
    df_model = df.dropna(subset=feat_cols + ["y"]).copy()

    # Standardize features
    X = df_model[feat_cols].to_numpy()
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    # Indices
    sa2_codes = pd.Index(sorted(df_model["sa2_code"].unique()))
    sa2_idx = pd.Categorical(df_model["sa2_code"], categories=sa2_codes).codes.astype(int)

    y = df_model["y"].astype(int).to_numpy()
    return df_model, Xz, y, sa2_idx, len(sa2_codes), feat_cols, mu, sd, df

def fit_forecast():
    df_model, Xz, y, sa2_idx, n_sa2, feat_cols, mu, sd, df_full = build_dataset()
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
            draws=1000, tune=1000, chains=4, target_accept=0.9,
            random_seed=RANDOM_SEED, progressbar=True
        )

    # ---- Predict for next month using latest available features per SA2 ----
    latest_month = df_full["month"].max()
    base = (df_full[df_full["month"] == latest_month]
            .copy()
            .drop_duplicates(subset=["sa2_code"]))

    # Build standardized X for prediction (impute with training means if needed)
    Xp = base[feat_cols].to_numpy()
    Xp = np.where(np.isnan(Xp), mu, Xp)
    Xp_z = (Xp - mu) / sd  # shape: (M, k)

    # Align SA2 index with training set
    train_sa2 = pd.Index(sorted(df_model["sa2_code"].unique()))
    idx_map = {code: i for i, code in enumerate(train_sa2)}
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

    S = beta_s.shape[0]

    # Compute probabilities per SA2
    probs = np.zeros(len(base), dtype=float)
    for i, row in base.reset_index(drop=True).iterrows():
        idx = int(row["sa2_idx"])
        # (S, k) @ (k,) -> (S,)
        lin = beta_s @ Xp_z[i, :]
        if idx >= 0:
            lin = lin + a_sa2_s[:, idx]
        probs[i] = _expit(lin).mean()

    out = pd.DataFrame({
        "sa2_code": base["sa2_code"].values,
        "month": (latest_month + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp(),
        "price_pressure_prob": probs
    })
    out.to_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet", index=False)
    print("Wrote price-pressure forecast → data_stage/price_pressure_forecast_sa2.parquet")
    return idata

if __name__ == "__main__":
    fit_forecast()
