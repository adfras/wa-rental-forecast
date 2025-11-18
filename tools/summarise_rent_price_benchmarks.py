"""Join rent and house-price benchmark summaries for side-by-side reporting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


RENT_SUMMARY = Path("outputs/v3/v3_benchmarks_summary.csv")
PRICE_SUMMARY = Path("outputs/v3_price/v3_price_benchmarks_summary.csv")
OUT_PATH = Path("outputs/tables/rent_price_benchmark_comparison.csv")


def main() -> Path:
    if not RENT_SUMMARY.exists():
        raise SystemExit(f"Rent benchmark summary missing: {RENT_SUMMARY}")
    if not PRICE_SUMMARY.exists():
        raise SystemExit(f"Price benchmark summary missing: {PRICE_SUMMARY}")

    rent = pd.read_csv(RENT_SUMMARY)
    price = pd.read_csv(PRICE_SUMMARY)

    merged = rent.merge(
        price,
        on=["model", "threshold", "decision_cut"],
        suffixes=("_rent", "_price"),
    )
    for metric in ("mean_precision", "mean_recall", "mean_accuracy"):
        merged[f"{metric}_delta_price_minus_rent"] = (
            merged[f"{metric}_price"] - merged[f"{metric}_rent"]
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(merged)} rows)")
    return OUT_PATH


if __name__ == "__main__":
    main()

