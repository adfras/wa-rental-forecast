"""Produce simple WA-level monthly summaries via DuckDB.

Writes outputs/tables/wa_monthly_prob_summary.csv with columns:
  - month
  - wa_avg_prob
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.common.duck import ddb_query, scan_stage
from src.config import OUT_DIR


def main():
    src = 'price_pressure_forecast_sa2_history.parquet'
    q = (
        "SELECT month, AVG(price_pressure_prob) AS wa_avg_prob "
        f"FROM {scan_stage(src)} GROUP BY month ORDER BY month"
    )
    df = ddb_query(q)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / 'wa_monthly_prob_summary.csv'
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
