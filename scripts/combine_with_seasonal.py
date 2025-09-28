"""Blend walk-forward forecast probabilities with a seasonal naive classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import STAGE_DIR, RENT_GROWTH_THRESHOLD


def load_history() -> pd.DataFrame:
    return pd.read_parquet(STAGE_DIR / "price_pressure_forecast_sa2_history.parquet")


def build_naive_prob() -> pd.DataFrame:
    bonds = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    bonds["month"] = pd.to_datetime(bonds["month"]).dt.to_period("M").dt.to_timestamp()
    bonds = bonds.sort_values(["sa2_code", "month"])
    bonds["rent_prev"] = bonds.groupby("sa2_code")["median_rent"].shift(1)
    bonds["actual_jump"] = (
        (bonds["median_rent"] / bonds["rent_prev"] - 1.0) > RENT_GROWTH_THRESHOLD
    ).astype(float)
    bonds["month"] = bonds["month"] + pd.offsets.DateOffset(months=12)
    naive = bonds.dropna(subset=["actual_jump"])[["sa2_code", "month", "actual_jump"]]
    naive = naive.rename(columns={"actual_jump": "naive_prob"})
    return naive


def blend(history: pd.DataFrame, naive: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = history.merge(naive, on=["sa2_code", "month"], how="left")
    df = df.sort_values(["sa2_code", "month"]).copy()
    df["naive_prob"] = df.groupby("month")["naive_prob"].transform(
        lambda s: s.fillna(s.mean())
    )
    df["naive_prob"] = df["naive_prob"].fillna(df["price_pressure_prob"].mean())
    df["price_pressure_prob"] = (
        weight * df["price_pressure_prob"] + (1.0 - weight) * df["naive_prob"]
    )
    df.drop(columns=["naive_prob"], inplace=True)
    return df


def main(weight: float, output: Path) -> None:
    hist = load_history()
    naive = build_naive_prob()
    blended = blend(hist, naive, weight)
    blended.to_parquet(output, index=False)
    print(f"wrote {len(blended)} rows to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Blend model probabilities with seasonal naive")
    ap.add_argument("--weight", type=float, default=0.7, help="Weight on model probabilities (0..1)")
    ap.add_argument("--output", type=Path, default=STAGE_DIR / "price_pressure_forecast_sa2_history.parquet")
    args = ap.parse_args()
    main(args.weight, args.output)
