"""Compute forward house-price growth labels for SA2 monthly series.

Usage:
    python -m tools.build_house_price_labels --thresholds 0.02 0.03

Outputs:
    data_stage/house_prices_sa2_labels.parquet with future gain columns and
    boolean jump indicators per requested threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import STAGE_DIR


def compute_labels(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    df = df.copy()
    df["sa2_code"] = df["sa2_code"].astype(str)
    df.sort_values(["sa2_code", "month"], inplace=True)

    group = df.groupby("sa2_code", sort=False)
    next_price = group["median_house_price"].shift(-1)
    df["future_price_gain"] = (next_price / df["median_house_price"]) - 1.0
    df["future_price_gain"] = df["future_price_gain"].replace([np.inf, -np.inf], np.nan)

    for thr in thresholds:
        col = f"price_jump_ge_{thr:.3f}".replace(".", "p")
        df[col] = (df["future_price_gain"] >= thr).astype("Int8")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate house-price growth labels")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(STAGE_DIR) / "house_prices_sa2_monthly.parquet",
        help="Input SA2 monthly house price parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(STAGE_DIR) / "house_prices_sa2_labels.parquet",
        help="Output parquet with future gains and jump indicators",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.02],
        help="Price growth thresholds to flag (default 0.02)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"House price source not found: {args.input}")

    df = pd.read_parquet(args.input)
    if "month" not in df.columns or "sa2_code" not in df.columns or "median_house_price" not in df.columns:
        raise SystemExit("Input parquet must contain sa2_code, month, median_house_price")

    df["month"] = pd.to_datetime(df["month"])
    labelled = compute_labels(df, thresholds=args.thresholds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(args.output, index=False)
    months = sorted({m.strftime("%Y-%m") for m in labelled["month"].dropna().unique()})
    print(
        f"Wrote {len(labelled)} house-price label rows → {args.output} "
        f"(thresholds={args.thresholds}, months={months[:3]}{'...' if len(months) > 3 else ''})"
    )


if __name__ == "__main__":
    main()

