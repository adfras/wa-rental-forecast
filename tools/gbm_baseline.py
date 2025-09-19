"""
Gradient-boosted baseline (LightGBM) for price-pressure probability and simple export.

This is an optional, non-blocking baseline. It requires external deps:
  pip install lightgbm scikit-learn

It reuses the same feature frame as src/models/forecast.py (availability_rate,
churn_rate, rent_mom_1m, rent_mom_3m) and trains a classifier on all
historical months with realized labels, then predicts for the next month.

Outputs:
  - outputs/gbm_baseline_latest.parquet (sa2_code, month, p_gbm)

Usage:
  # Ensure nowcast exists (availability_rate)
  python -m src.models.nowcast --draws 400 --tune 400 --chains 1 --cores 1
  # Then run baseline
  python -m tools.gbm_baseline

Notes:
  - If LightGBM is unavailable, the script prints a hint and exits gracefully.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import STAGE_DIR, OUT_DIR, RENT_GROWTH_THRESHOLD
from src.features.dates import to_month
from src.features.engineering import add_churn_proxy, add_rent_momentum


def _warn(msg: str):
    print(f"[gbm] {msg}")


def _load_feature_frame() -> pd.DataFrame:
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    now = pd.read_parquet(STAGE_DIR / "availability_nowcast_sa2.parquet").copy()
    sa2["month"] = to_month(sa2["month"])
    now["month"] = to_month(now["month"])
    df = sa2.merge(now, on=["sa2_code", "month"], how="left")
    df = add_rent_momentum(df, group_col="sa2_code", month_col="month", rent_col="median_rent")
    df = add_churn_proxy(df)
    # Label: next-month rent rise > threshold
    df["median_rent_next"] = df.groupby("sa2_code")["median_rent"].shift(-1)
    df["y"] = ((df["median_rent_next"] - df["median_rent"]) / df["median_rent"] > RENT_GROWTH_THRESHOLD).astype(float)
    return df


def main() -> None:
    try:
        import lightgbm as lgb
    except Exception:
        _warn("LightGBM not installed. Run: pip install lightgbm scikit-learn")
        return

    try:
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split
    except Exception:
        _warn("scikit-learn not installed. Run: pip install scikit-learn")
        return

    df = _load_feature_frame()
    feat_cols = ["availability_rate", "churn_rate", "rent_mom_1m", "rent_mom_3m"]
    use = df.dropna(subset=feat_cols + ["y"]).copy()
    # Train on all but last month to emulate live inference
    latest_m = use["month"].max()
    train = use[use["month"] < latest_m].copy()
    base = use[use["month"] == latest_m].copy()
    if train.empty or base.empty:
        _warn("Insufficient data for baseline training or base month prediction.")
        return

    X = train[feat_cols].to_numpy()
    y = train["y"].astype(int).to_numpy()
    Xp = base[feat_cols].to_numpy()

    # Basic LightGBM binary classifier
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=20,
        verbose=-1,
        seed=2025,
    )
    dtrain = lgb.Dataset(X, label=y)
    bst = lgb.train(params, dtrain, num_boost_round=400)
    p = bst.predict(Xp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "sa2_code": base["sa2_code"].astype(str).values,
        "month": (latest_m + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp(),
        "p_gbm": p.astype(float),
    })
    out.to_parquet(OUT_DIR / "gbm_baseline_latest.parquet", index=False)
    print(f"[gbm] Wrote {OUT_DIR / 'gbm_baseline_latest.parquet'} ({len(out)} rows for {(latest_m + pd.offsets.MonthBegin(1)).strftime('%Y-%m')})")


if __name__ == "__main__":
    main()
