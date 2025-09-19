"""
Export Top-20 SA2 alerts for a given month at a chosen threshold.

Usage:
  python -m tools.export_alerts --thr 0.34            # uses latest month
  python -m tools.export_alerts --thr 0.30 --month 2025-09

Writes outputs/tables/alerts_top20_<month>_thr_<thr*1000|int>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(thr: float, month: str | None) -> Path:
    df = pd.read_parquet('data_stage/price_pressure_forecast_sa2.parquet')
    df['month'] = pd.to_datetime(df['month']).dt.to_period('M').dt.to_timestamp()
    if month:
        m = pd.to_datetime(month).to_period('M').to_timestamp()
    else:
        m = df['month'].max()
    cur = df[df['month'] == m].copy()
    if cur.empty:
        raise SystemExit(f'No rows for month {m:%Y-%m}.')
    cur['alert'] = (cur['price_pressure_prob'] >= float(thr)).astype(int)

    # Join SA2 names if available
    try:
        from src.reporting.evaluate_forecasts import _read_sa2_names  # reuse util
        names = _read_sa2_names()
        cur = cur.merge(names, on='sa2_code', how='left')
    except Exception:
        pass

    top = cur.sort_values('price_pressure_prob', ascending=False).head(20)
    outdir = Path('outputs/tables'); outdir.mkdir(parents=True, exist_ok=True)
    tag_thr = str(int(round(thr * 1000)))  # e.g., 0.34 -> 340
    out = outdir / f"alerts_top20_{m:%Y-%m}_thr_{tag_thr}.csv"
    cols = ['sa2_code'] + ([c for c in ['sa2_name'] if c in top.columns]) + ['month','price_pressure_prob','alert']
    top[cols].to_csv(out, index=False)
    print(f'Wrote {out} (thr={thr})')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=float, required=True)
    ap.add_argument('--month', type=str, default=None)
    args = ap.parse_args()
    main(args.thr, args.month)
