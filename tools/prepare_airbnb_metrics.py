#!/usr/bin/env python
"""Aggregate Airbnb listing-level data to SA2-month metrics for forecast v3.

Input CSV should include columns:
    sa2_code, month, listing_id, daily_rate, occupancy_rate, stays_per_month,
    review_score, is_superhost, bedrooms, accommodates

Additional columns are allowed; aggregation rules can be extended easily.

Usage:
    python tools/prepare_airbnb_metrics.py data_raw/airbnb_listings.csv

Writes:
    data_stage/airbnb_metrics.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import STAGE_DIR

AGG_MAP = {
    "listing_id": "count",
    "daily_rate": "mean",
    "occupancy_rate": "mean",
    "stays_per_month": "mean",
    "review_score": "mean",
    "is_superhost": "mean",
    "bedrooms": "mean",
    "accommodates": "mean",
}


def main(path: str) -> None:
    csv_path = Path(path)
    if not csv_path.exists():
        raise SystemExit(f"Input file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"sa2_code", "month"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise SystemExit(f"CSV missing required columns: {missing}")
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    grouped = df.groupby(["sa2_code", "month"], as_index=False).agg(AGG_MAP)
    grouped = grouped.rename(columns={
        "listing_id": "listing_count",
        "daily_rate": "avg_daily_rate",
        "occupancy_rate": "occupancy_rate",
        "stays_per_month": "average_stays_per_month",
        "review_score": "mean_review_score",
        "is_superhost": "share_superhost",
        "bedrooms": "avg_bedrooms",
        "accommodates": "avg_accommodates",
    })
    out_path = Path(STAGE_DIR) / "airbnb_metrics.parquet"
    grouped.to_parquet(out_path, index=False)
    print(f"Wrote Airbnb metrics → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/prepare_airbnb_metrics.py <path-to-csv>")
    main(sys.argv[1])
