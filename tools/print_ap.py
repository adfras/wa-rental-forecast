"""
Compute Average Precision (AP) for specified months using the evaluation details
CSVs produced by src.reporting.evaluate_forecasts. Usage:

  python -m tools.print_ap --months 2025-07 2025-08
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sklearn.metrics import average_precision_score


def ap_for_month(month: str) -> float | None:
    path = Path("outputs/evaluations") / f"forecast_eval_details_{month}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    y = df["actual_jump"].astype(int).to_numpy()
    p = df["price_pressure_prob"].astype(float).to_numpy()
    return float(average_precision_score(y, p))


def main(months: list[str]) -> None:
    any_ok = False
    for m in months:
        ap = ap_for_month(m)
        if ap is None:
            print(f"{m}: (no details CSV found)")
        else:
            any_ok = True
            print(f"AP {m}: {ap:.3f}")
    if not any_ok:
        print("No AP computed; run evaluation to generate details CSVs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs='+', required=True)
    args = ap.parse_args()
    main(args.months)
