"""
Hierarchical logistic forecast of next-month rent rise probability by SA2.

Features include nowcasted availability_rate, churn, and rent momentum. The
model includes SA2 random intercepts. Writes:
 - data_stage/price_pressure_forecast_sa2.parquet
 - appends to data_stage/price_pressure_forecast_sa2_history.parquet
"""
import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import argparse
import os

from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import EVAL_DIR, STAGE_DIR, RENT_GROWTH_THRESHOLD, RANDOM_SEED
from src.features.dates import compute_recency_weights, to_month
from src.features.engineering import (
    add_calendar_features,
    add_churn_proxy,
    add_rent_momentum,
    apply_standardization,
    add_group_demean,
    add_interaction_features,
    standardize_columns,
)

def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_dataset():
    """Prepare merged features and labels for the logistic model."""
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    now = pd.read_parquet(STAGE_DIR / "availability_nowcast_sa2.parquet").copy()

    # Align month types
    sa2["month"] = to_month(sa2["month"])
    now["month"] = to_month(now["month"])

    df = sa2.merge(now, on=["sa2_code", "month"], how="left")

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
    df["y"] = ((df["median_rent_next"] - df["median_rent"]) / df["median_rent"] > RENT_GROWTH_THRESHOLD).astype(float)

    # Calendar + simple lags
    df = add_calendar_features(df)
    df = df.sort_values(["sa2_code","month"]).copy()
    df["availability_rate_lag1"] = df.groupby("sa2_code")["availability_rate"].shift(1)
    df["churn_rate_lag1"] = df.groupby("sa2_code")["churn_rate"].shift(1)
    df["rent_mom_12m"] = (df["median_rent"] / df.groupby("sa2_code")["median_rent"].shift(12)) - 1.0

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
    for col in demean_cols:
        name = f"{col}_wa_dev"
        if name in df.columns:
            feat_cols.append(name)
    for a, b in interaction_pairs:
        name = f"{a}__x__{b}"
        if name in df.columns:
            feat_cols.append(name)
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
    df_model = df.dropna(subset=feat_cols + ["y"]).copy()

    # Standardize features
    Xz, mu, sd = standardize_columns(df_model, feat_cols)

    # Indices
    sa2_codes = pd.Index(sorted(df_model["sa2_code"].unique()))
    sa2_idx = pd.Categorical(df_model["sa2_code"], categories=sa2_codes).codes.astype(int)

    y = df_model["y"].astype(int).to_numpy()
    return df_model, Xz, y, sa2_idx, sa2_codes, feat_cols, mu, sd, df

def fit_forecast(draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4,
                 recency_half_life: float | None = None,
                 bias_correct_l6: bool = False,
                 calibrate_isotonic: bool = False,
                 prior_shift: bool = False,
                 auto_calibrate: bool = True,
                 target_accept: float = 0.92):
    """Fit the hierarchical logistic forecast and write predictions + history."""
    df_model, Xz, y, sa2_idx, sa2_codes, feat_cols, mu, sd, df_full = build_dataset()
    n_sa2 = len(sa2_codes)
    n, k = Xz.shape

    # Indices for key features to enable mild varying slopes
    idx_avail = feat_cols.index("availability_rate") if "availability_rate" in feat_cols else None
    idx_churn = feat_cols.index("churn_rate") if "churn_rate" in feat_cols else None

    # Optional recency weights (half-life in months)
    w = compute_recency_weights(df_model["month"], recency_half_life)

    with pm.Model() as m:
        # Tighter prior on coefficients to reduce extreme logits and improve calibration
        beta = pm.Normal("beta", 0.0, 0.5, shape=k)

        mu_a = pm.Normal("mu_a", 0.0, 1.0)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_sa2)
        a_sa2 = pm.Deterministic("a_sa2", mu_a + z_a * sigma_a)

        # Mild hierarchical varying slopes for key drivers (centered at 0)
        b_av_sa2 = None
        if idx_avail is not None:
            mu_b_av = pm.Normal("mu_b_av", 0.0, 0.2)
            sigma_b_av = pm.HalfNormal("sigma_b_av", 0.5)
            z_b_av = pm.Normal("z_b_av", 0.0, 1.0, shape=n_sa2)
            b_av_sa2 = pm.Deterministic("b_av_sa2", mu_b_av + z_b_av * sigma_b_av)

        b_ch_sa2 = None
        if idx_churn is not None:
            mu_b_ch = pm.Normal("mu_b_ch", 0.0, 0.2)
            sigma_b_ch = pm.HalfNormal("sigma_b_ch", 0.5)
            z_b_ch = pm.Normal("z_b_ch", 0.0, 1.0, shape=n_sa2)
            b_ch_sa2 = pm.Deterministic("b_ch_sa2", mu_b_ch + z_b_ch * sigma_b_ch)

        eta = a_sa2[sa2_idx] + pm.math.dot(Xz, beta)
        if idx_avail is not None:
            eta = eta + b_av_sa2[sa2_idx] * Xz[:, idx_avail]
        if idx_churn is not None:
            eta = eta + b_ch_sa2[sa2_idx] * Xz[:, idx_churn]
        if w is None:
            pm.Bernoulli("y", logit_p=eta, observed=y)
        else:
            y_dist = pm.Bernoulli.dist(logit_p=eta)
            pm.Potential("weighted_loglik_binom", (w * pm.logp(y_dist, y)).sum())

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept,
            random_seed=RANDOM_SEED, progressbar=True
        )

    # ---- Predict for next month using latest available features per SA2 ----
    latest_month = df_full["month"].max()
    base = (df_full[df_full["month"] == latest_month]
            .copy()
            .drop_duplicates(subset=["sa2_code"]))

    # Build standardized X for prediction (impute with training means if needed)
    Xp_z = apply_standardization(base, feat_cols, mu, sd)

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

    def _posterior_summary(X_rows: np.ndarray, sa2_indices: np.ndarray, *, quantiles: bool = False):
        X_rows = np.asarray(X_rows)
        sa2_indices = np.asarray(sa2_indices, dtype=int)
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
    train_prob_mean = _posterior_summary(Xz, sa2_idx, quantiles=False)

    # Forecast probabilities with credible intervals
    base_probs, prob_p05, prob_p50, prob_p95 = _posterior_summary(
        Xp_z, base["sa2_idx"].to_numpy(), quantiles=True
    )

    out = pd.DataFrame({
        "sa2_code": base["sa2_code"].values,
        "month": (latest_month + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp(),
        "price_pressure_prob": base_probs,
        "prob_p05": prob_p05,
        "prob_p50": prob_p50,
        "prob_p95": prob_p95,
    })
    prior_shift_applied = False
    bias_correct_applied = False
    calibration_applied = False
    # Optional prior-shift (label-prevalence) adjustment of logits toward recent base rate
    if prior_shift or str(os.getenv("PRIOR_SHIFT", "0")).lower() in {"1","true","yes"}:
        try:
            # Training prevalence
            pi_train = float(y.mean()) if len(y) else None
            # Estimate recent realized prevalence over last K months
            Kp = int(os.getenv("PRIOR_MONTHS", "3"))
            sa2_full = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
            sa2_full["month"] = pd.to_datetime(sa2_full["month"]).dt.to_period("M").dt.to_timestamp()
            sa2_full = sa2_full.sort_values(["sa2_code","month"]).copy()
            sa2_full["rent_prev"] = sa2_full.groupby("sa2_code")["median_rent"].shift(1)
            from src.config import RENT_GROWTH_THRESHOLD as _TH
            sa2_full["actual_jump"] = (sa2_full["median_rent"] / sa2_full["rent_prev"] - 1.0) > _TH
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
    should_calibrate = calibrate_isotonic or env_calibrate
    if auto_calibrate and not should_calibrate:
        if df_model["month"].nunique() >= 6 and df_model["y"].sum() >= 20:
            should_calibrate = True
    if should_calibrate:
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore
            # Build training pairs from history joined with realized labels
            hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
            hist = pd.read_parquet(hist_path)
            sa2_full = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
            sa2_full["month"] = pd.to_datetime(sa2_full["month"]).dt.to_period("M").dt.to_timestamp()
            sa2_full = sa2_full.sort_values(["sa2_code","month"]).copy()
            sa2_full["rent_prev"] = sa2_full.groupby("sa2_code")["median_rent"].shift(1)
            from src.config import RENT_GROWTH_THRESHOLD as _TH
            sa2_full["actual_jump"] = (sa2_full["median_rent"] / sa2_full["rent_prev"] - 1.0) > _TH
            realized = sa2_full.loc[sa2_full["rent_prev"].notna(), ["sa2_code","month","actual_jump"]]
            j = (hist.merge(realized, on=["sa2_code","month"], how="inner")
                    .sort_values(["month"]))
            Kc = int(os.getenv("CALIB_MONTHS", "6"))
            if not j.empty:
                # Use last Kc months where both pred and actual exist
                jm = j.groupby("month").size().reset_index(name="n")
                last_months = jm["month"].sort_values().tail(max(Kc,1)).tolist()
                cal = j[j["month"].isin(last_months)].copy()
                p_tr = cal["price_pressure_prob"].astype(float).to_numpy()
                y_tr = cal["actual_jump"].astype(int).to_numpy()
                if p_tr.size >= 10 and len(set(y_tr)) == 2:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(p_tr, y_tr)
                    out["price_pressure_prob_raw"] = out["price_pressure_prob"]
                    out["price_pressure_prob"] = iso.transform(out["price_pressure_prob"].to_numpy())
                    calibration_applied = True
        except Exception:
            pass
    # Optional per-SA2 bias correction using last 6 realized months
    if bias_correct_l6 or str(os.getenv("BIAS_CORRECT_L6", "0")).lower() in {"1","true","yes"}:
        try:
            # Load history (if any) and realized outcomes
            hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
            try:
                hist = pd.read_parquet(hist_path)
            except FileNotFoundError:
                hist = pd.DataFrame(columns=["sa2_code","month","price_pressure_prob"])
            sa2_full = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
            sa2_full["month"] = pd.to_datetime(sa2_full["month"]).dt.to_period("M").dt.to_timestamp()
            sa2_full = sa2_full.sort_values(["sa2_code","month"]).copy()
            sa2_full["rent_prev"] = sa2_full.groupby("sa2_code")["median_rent"].shift(1)
            # label using configured threshold
            from src.config import RENT_GROWTH_THRESHOLD as _TH
            sa2_full["actual_jump"] = (sa2_full["median_rent"] / sa2_full["rent_prev"] - 1.0) > _TH
            realized = sa2_full.loc[sa2_full["rent_prev"].notna(), ["sa2_code","month","actual_jump"]]
            # Join, take last K months per SA2
            joined = (hist.merge(realized, on=["sa2_code","month"], how="inner")
                           .sort_values(["sa2_code","month"]))
            K = int(os.getenv("BIAS_CORR_MONTHS", "6"))
            gamma = float(os.getenv("BIAS_CORR_GAMMA", "0.25"))
            # compute mean residual p - y over last K months per SA2
            joined["resid"] = joined["price_pressure_prob"].astype(float) - joined["actual_jump"].astype(int)
            resid = (joined.groupby("sa2_code", group_keys=False)
                           .apply(lambda d: d.tail(max(K, 1))["resid"].mean())
                           .to_frame("resid_mean").reset_index())
            out = out.merge(resid, on="sa2_code", how="left")
            out["price_pressure_prob_raw"] = out["price_pressure_prob"]
            out["price_pressure_prob"] = np.clip(out["price_pressure_prob"] - gamma*out["resid_mean"].fillna(0.0), 0.0, 1.0)
            out = out.drop(columns=["resid_mean"])  # keep raw as separate column
            bias_correct_applied = True
        except Exception:
            # Fail safe: skip bias correction on any error
            pass
    # --- Training-set diagnostics & recommended threshold ---
    metrics = {
        "train_rows": int(len(y)),
        "train_positive": int(y.sum()),
        "train_base_rate": float(y.mean()) if len(y) else float("nan"),
        "train_auc": float(roc_auc_score(y, train_prob_mean)) if len(np.unique(y)) > 1 else float("nan"),
        "train_average_precision": float(average_precision_score(y, train_prob_mean)) if len(np.unique(y)) > 1 else float("nan"),
    }

    threshold_grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    threshold_stats = []
    for thr in threshold_grid:
        preds = (train_prob_mean >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        tn = int(np.sum((preds == 0) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        accuracy = (tp + tn) / max(len(y), 1)
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
    metrics_path = EVAL_DIR / "forecast_training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out.to_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet", index=False)
    print("Wrote price-pressure forecast → data_stage/price_pressure_forecast_sa2.parquet")
    
    # ---- maintain an append-only history for the website time slider ----
    hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
    try:
        old = pd.read_parquet(hist_path)
        combined = (pd.concat([old, out], ignore_index=True)
                    .drop_duplicates(subset=["sa2_code", "month"])
                    .sort_values(["sa2_code", "month"]))
    except FileNotFoundError:
        combined = out.copy()

    combined.to_parquet(hist_path, index=False)
    print(f"Updated forecast history → {hist_path} (months={combined['month'].nunique()})")
    
    
    return idata

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit hierarchical logistic forecast and write predictions + history")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--recency-half-life", type=float, default=None,
                    help="Half-life in months for sample recency weights (e.g., 12 → 50% per year older)")
    ap.add_argument("--bias-correct-l6", action="store_true",
                    help="Apply per-SA2 bias correction using last 6 realized months (env BIAS_CORR_MONTHS to change window)")
    ap.add_argument("--calibrate-isotonic", action="store_true",
                    help="Apply isotonic calibration using last K realized months (env CALIB_MONTHS)")
    ap.add_argument("--prior-shift", action="store_true",
                    help="Adjust logits toward recent realized prevalence (env PRIOR_MONTHS; default 3)")
    ap.add_argument("--target-accept", type=float, default=0.92,
                    help="Target acceptance rate for the NUTS sampler (default 0.92).")
    ap.add_argument("--no-auto-calibrate", dest="auto_calibrate", action="store_false",
                    help="Disable automatic isotonic calibration when sufficient history exists.")
    ap.set_defaults(auto_calibrate=True)
    args = ap.parse_args()
    fit_forecast(draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
                 recency_half_life=args.recency_half_life, bias_correct_l6=args.bias_correct_l6,
                 calibrate_isotonic=args.calibrate_isotonic, prior_shift=args.prior_shift,
                 auto_calibrate=args.auto_calibrate, target_accept=args.target_accept)
