"""
Export Top-20 SA2 alerts using the per-month best_thr_f1 from evaluation summary.

Usage:
  python -m tools.export_alerts_best                   # uses latest month with summary
  python -m tools.export_alerts_best --month 2025-08   # export for a specific month

Writes outputs/tables/alerts_top20_<month>_best.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(month: str | None = None) -> Path:
    summ_p = Path('outputs/evaluations/forecast_eval_summary.csv')
    preds_p = Path('data_stage/price_pressure_forecast_sa2.parquet')
    if not summ_p.exists():
        raise SystemExit('Missing outputs/evaluations/forecast_eval_summary.csv. Run evaluation first.')
    if not preds_p.exists():
        raise SystemExit('Missing data_stage/price_pressure_forecast_sa2.parquet. Fit forecast first.')

    summ = pd.read_csv(summ_p, parse_dates=['month'])
    summ['month'] = pd.to_datetime(summ['month']).dt.to_period('M').dt.to_timestamp()
    preds = pd.read_parquet(preds_p)
    preds['month'] = pd.to_datetime(preds['month']).dt.to_period('M').dt.to_timestamp()

    if month:
        m = pd.to_datetime(month).to_period('M').to_timestamp()
        row = summ[summ['month'] == m]
        if row.empty:
            raise SystemExit(f'No summary row for month {m:%Y-%m}.')
    else:
        # latest month present in both preds and summary
        common = sorted(set(summ['month'].unique()).intersection(set(preds['month'].unique())))
        if not common:
            raise SystemExit('No common months between predictions and summary.')
        m = common[-1]
        row = summ[summ['month'] == m]

    thr = float(row['best_thr_f1'].iloc[0]) if 'best_thr_f1' in row.columns else 0.5
    cur = preds[preds['month'] == m].copy()
    cur['alert'] = (cur['price_pressure_prob'] >= thr).astype(int)

    # Join SA2 names if available
    try:
        from src.reporting.evaluate_forecasts import _read_sa2_names  # reuse utility
        names = _read_sa2_names()
        cur = cur.merge(names, on='sa2_code', how='left')
    except Exception:
        pass

    top = cur.sort_values('price_pressure_prob', ascending=False).head(20)
    outdir = Path('outputs/tables'); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"alerts_top20_{m:%Y-%m}_best.csv"
    cols = ['sa2_code'] + ([c for c in ['sa2_name'] if c in top.columns]) + ['month','price_pressure_prob','alert']
    top[cols].to_csv(out, index=False)
    print(f'Wrote {out} (thr={thr:.3f})')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', type=str, default=None)
    args = ap.parse_args()
    main(args.month)
