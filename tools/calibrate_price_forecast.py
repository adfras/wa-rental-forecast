"""Calibrate house-price jump probabilities using isotonic regression."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

from src.config import STAGE_DIR


def load_price_forecast(threshold: float) -> pd.DataFrame:
    path = Path(STAGE_DIR) / "price_pressure_forecast_sa2_history.parquet"
    if not path.exists():
        raise SystemExit(f"Forecast history not found: {path}")
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    mask = np.isclose(df.get("rent_jump_threshold", threshold), threshold)
    df = df[mask].copy()
    if df.empty:
        raise SystemExit(f"No forecasts for threshold {threshold:.3f} in {path}")
    return df


def compute_price_labels(threshold: float) -> pd.DataFrame:
    path = Path(STAGE_DIR) / "house_prices_sa2_monthly.parquet"
    if not path.exists():
        raise SystemExit(f"House price data missing: {path}")
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    df = df.sort_values(["sa2_code", "month"])
    df["median_house_price_prev"] = df.groupby("sa2_code")["median_house_price"].shift(1)
    df["price_gain"] = df["median_house_price"] / df["median_house_price_prev"] - 1.0
    df["label"] = (df["price_gain"] > threshold).astype(int)
    df = df.dropna(subset=["median_house_price_prev"])  # drop first month per SA2
    df = df[["sa2_code", "month", "label", "price_gain"]]
    return df


def calibrate(threshold: float, output: Path | None) -> Path:
    forecasts = load_price_forecast(threshold)
    labels = compute_price_labels(threshold)
    merged = forecasts.merge(labels, on=["sa2_code", "month"], how="left")

    train_mask = merged["label"].notna()
    train = merged[train_mask].copy()
    if train["label"].nunique() < 2:
        raise SystemExit("Not enough positive/negative outcomes to calibrate.")

    probs = train["price_pressure_prob"].astype(float).to_numpy()
    labels_bin = train["label"].astype(int).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels_bin)

    merged["price_pressure_prob_calib"] = iso.transform(merged["price_pressure_prob"].astype(float))

    probs_clip = np.clip(probs, 1e-6, 1 - 1e-6)
    probs_calib = np.clip(merged.loc[train_mask, "price_pressure_prob_calib"], 1e-6, 1 - 1e-6)
    logloss_before = log_loss(labels_bin, probs_clip, labels=[0, 1])
    logloss_after = log_loss(labels_bin, probs_calib, labels=[0, 1])
    brier_before = brier_score_loss(labels_bin, probs_clip)
    brier_after = brier_score_loss(labels_bin, probs_calib)

    out_path = output or (Path(STAGE_DIR) / "price_pressure_forecast_sa2_price_calibrated.parquet")
    merged.to_parquet(out_path, index=False)

    print(f"Training rows: {len(train)} (positives={train['label'].sum()}, negatives={len(train)-train['label'].sum()})")
    print(f"Log-loss: {logloss_before:.4f} → {logloss_after:.4f}")
    print(f"Brier score: {brier_before:.4f} → {brier_after:.4f}")
    print(f"Calibrated forecasts written to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate house-price jump forecast")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    calibrate(args.threshold, args.output)


if __name__ == "__main__":
    main()

