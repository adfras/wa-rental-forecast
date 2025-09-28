"""
Tree-based (GBM) forecast of next-month rent rise probability by SA2.

Rationale: non-parametric, no MCMC → no R-hat concerns; strong at capturing
nonlinearities and interactions. We reuse the same feature builder as the
Bayesian model and emit identical artifacts for downstream steps.

Outputs:
 - data_stage/price_pressure_forecast_sa2.parquet
 - append/update data_stage/price_pressure_forecast_sa2_history.parquet
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from src.config import STAGE_DIR, EVAL_DIR, RANDOM_SEED
from src.models.forecast import build_dataset
from src.features.dates import compute_recency_weights


@dataclass
class GBMSettings:
    learning_rate: float = 0.05
    n_estimators: int = 800
    max_depth: int = 3
    min_samples_leaf: int = 20


def _fit_gbm(X: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: GBMSettings):
    gbm = GradientBoostingClassifier(
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=RANDOM_SEED,
    )
    gbm.fit(X, y, sample_weight=w)
    return gbm


def _calibrate_isotonic(probs: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    try:
        if len(np.unique(y)) < 2:
            return None
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, y)
        return iso
    except Exception:
        return None


def fit_forecast_gbm(draws: int = 0, tune: int = 0,  # unused, kept for CLI symmetry
                     recency_half_life: float | None = 12,
                     calibrate_isotonic: bool = True) -> None:
    df_model, Xz, y, sa2_idx, sa2_codes, feat_cols, mu, sd, df_full, _, _ = build_dataset()

    # Use raw (imputed) feature matrix for trees (avoid clipping/standardization)
    X_df = df_model[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.mean())
    X = X_df.to_numpy(dtype=float)
    w = compute_recency_weights(df_model["month"], recency_half_life)
    if w is not None:
        w = w.astype(float)

    gbm = _fit_gbm(X, y.astype(int), w, GBMSettings())
    p_train = gbm.predict_proba(X)[:, 1]
    iso = _calibrate_isotonic(p_train, y) if calibrate_isotonic else None

    # Predict for next month
    latest_month = df_full["month"].max()
    base = (df_full[df_full["month"] == latest_month]
            .copy()
            .drop_duplicates(subset=["sa2_code"]))
    Xb = base.reindex(columns=feat_cols)
    Xb = Xb.apply(pd.to_numeric, errors="coerce").fillna(X_df.mean()).to_numpy(dtype=float)
    p = gbm.predict_proba(Xb)[:, 1]
    if iso is not None:
        try:
            p_raw = p.copy()
            p = iso.transform(p)
        except Exception:
            p_raw = None
    else:
        p_raw = None

    out = pd.DataFrame({
        "sa2_code": base["sa2_code"].astype(str).values,
        "month": (latest_month + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp(),
        "price_pressure_prob": p,
    })
    if p_raw is not None:
        out["price_pressure_prob_raw"] = p_raw

    out.to_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet", index=False)
    print("[gbm] Wrote price-pressure forecast → data_stage/price_pressure_forecast_sa2.parquet")

    # Update history
    hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
    try:
        old = pd.read_parquet(hist_path)
        combined = (pd.concat([old, out], ignore_index=True)
                    .drop_duplicates(subset=["sa2_code", "month"], keep="last")
                    .sort_values(["sa2_code", "month"]))
    except FileNotFoundError:
        combined = out.copy()
    combined.to_parquet(hist_path, index=False)
    print(f"[gbm] Updated forecast history → {hist_path} (months={combined['month'].nunique()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GBM forecast (no MCMC)")
    ap.add_argument("--recency-half-life", type=float, default=12)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.set_defaults(calibrate=True)
    args = ap.parse_args()
    fit_forecast_gbm(recency_half_life=args.recency_half_life,
                     calibrate_isotonic=args.calibrate)
