"""Generate expected-gain watchlists for rent or house-price forecasts.

Usage examples
--------------
Rent (default threshold 0.02):
    python -m tools.export_expected_gain --target rent --top 20

House price (same threshold):
    python -m tools.export_expected_gain --target price --top 20

Outputs go to ``outputs/tables/expected_gain_<target>_<YYYY-MM>.csv`` and
are also printed to stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.sa2 import load_sa2_names
from src.config import ASGS_SA2_GPKG, STAGE_DIR


def _forecast_path(target: str) -> Path:
    if target == "price":
        calibrated = Path(STAGE_DIR) / "price_pressure_forecast_sa2_price_calibrated.parquet"
        if calibrated.exists():
            return calibrated
    return Path(STAGE_DIR) / "price_pressure_forecast_sa2_history.parquet"


def _latest_forecast(target: str, threshold: float) -> pd.DataFrame:
    path = _forecast_path(target)
    if not path.exists():
        raise SystemExit(f"Forecast history not found: {path}")
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    mask = np.isclose(df.get("rent_jump_threshold", threshold), threshold)
    df = df[mask].copy()
    if df.empty:
        raise SystemExit(f"No rows found for threshold {threshold:.3f} in {path}")
    latest_month = df["month"].max()
    latest = df[df["month"] == latest_month].copy()
    latest.sort_values("price_pressure_prob", ascending=False, inplace=True)
    latest["target_month"] = latest_month
    return latest


def _mean_positive_jump(df: pd.DataFrame, value_col: str, threshold: float) -> pd.Series:
    df = df.copy()
    df.sort_values(["sa2_code", "month"], inplace=True)
    prev = df.groupby("sa2_code")[value_col].shift(1)
    jump = df[value_col] / prev - 1.0
    mask = (prev.notna()) & (jump > threshold)
    df = df.loc[mask, ["sa2_code"]].copy()
    df["jump"] = jump[mask]
    return df.groupby("sa2_code")["jump"].mean()


def _jump_table(target: str, threshold: float) -> pd.Series:
    if target == "rent":
        path = Path(STAGE_DIR) / "bonds_panel_sa2.parquet"
        value_col = "median_rent"
    else:
        path = Path(STAGE_DIR) / "house_prices_sa2_monthly.parquet"
        value_col = "median_house_price"
    if not path.exists():
        raise SystemExit(f"Required dataset missing: {path}")
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    return _mean_positive_jump(df[["sa2_code", "month", value_col]], value_col, threshold)


def export_expected_gain(target: str, threshold: float, top_n: int, output: Path | None) -> Path:
    latest = _latest_forecast(target, threshold)
    target_month = latest["month"].iloc[0]
    jump_series = _jump_table(target, threshold)
    global_mean = float(jump_series.mean()) if not jump_series.empty else 0.0
    prob_col = "price_pressure_prob_calib" if target == "price" and "price_pressure_prob_calib" in latest.columns else "price_pressure_prob"
    probs = latest[prob_col].astype(float)
    jumps = latest["sa2_code"].astype(str).map(jump_series).fillna(global_mean)
    latest["expected_gain"] = probs * jumps
    latest = latest.sort_values("expected_gain", ascending=False).head(top_n)
    latest["target_month"] = target_month.strftime("%Y-%m")
    latest["sa2_code"] = latest["sa2_code"].astype(str)
    names = load_sa2_names(ASGS_SA2_GPKG)
    latest = latest.merge(names, on="sa2_code", how="left")
    latest = latest[
        [
            "target_month",
            "sa2_code",
            "sa2_name",
            "price_pressure_prob",
            "expected_gain",
        ]
    ]
    outdir = Path("outputs") / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = output or (outdir / f"expected_gain_{target}_{target_month:%Y-%m}.csv")
    latest.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export expected-gain watchlist")
    parser.add_argument("--target", choices=["rent", "price"], default="price")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out = export_expected_gain(args.target, args.threshold, args.top, args.output)
    df = pd.read_csv(out)
    print(df.to_string(index=False))
    print(f"\nWrote {out} (target={args.target}, threshold={args.threshold:.3f})")


if __name__ == "__main__":
    main()
