#!/usr/bin/env python
"""Convert a CSV of macro indicators into the standard parquet format for forecast v3.

Expected columns (case-insensitive, extras allowed):
    month, nominal_gdp, disposable_income, cpi_housing, m2, policy_rate,
    vacancy_rate, inflation_expectations, unemployment_rate

Usage:
    python tools/prepare_macro_signals.py data_raw/macro_signals.csv

Writes:
    data_stage/macro_indicators.parquet
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "nominal_gdp": "macro_nominal_gdp",
        "gdp_nominal": "macro_nominal_gdp",
        "disposable_income": "macro_disposable_income",
        "cpi_housing": "macro_cpi_housing",
        "m2": "macro_m2",
        "policy_rate": "macro_policy_rate",
        "cash_rate": "macro_policy_rate",
        "vacancy_rate": "macro_vacancy_rate",
        "inflation_expectations": "macro_inflation_expectations",
        "unemployment_rate": "macro_unemployment_rate",
        "housing_price_index": "macro_house_price_index",
    }
    renamed = {}
    for col in df.columns:
        key = col.strip().lower()
        renamed[col] = col_map.get(key, col)
    df = df.rename(columns=renamed)
    return df


def main(path: str) -> None:
    csv_path = Path(path)
    if not csv_path.exists():
        raise SystemExit(f"Input file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "month" not in df.columns:
        raise SystemExit("CSV must contain a 'month' column (YYYY-MM or YYYY-MM-DD)")
    df = normalize_columns(df)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")
    out_path = Path(STAGE_DIR) / "macro_indicators.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote macro indicators → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/prepare_macro_signals.py <path-to-csv>")
    main(sys.argv[1])
