"""Quantify decision value from including median house prices at triage time.

Compares two policies for each realized month:
  A) Baseline: pick top-K SA2s by rent-risk probability p
  B) Price-aware: pick top-K by p * (hp_norm ** beta), where hp_norm = house_price / median(house_price)

Outputs per month and per K:
  - correct_at_k_baseline, correct_at_k_priceaware
  - value_hit_baseline, value_hit_priceaware (sum of y * house_price in top-K)
  - value_gain_abs, value_gain_pct

Usage:
  python -m tools.value_gain_from_houseprice --k 10 20 --beta 1.0 \
      --start 2025-03 --end 2025-08

Writes outputs/evaluations/value_gain_from_houseprice.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import STAGE_DIR, EVAL_DIR
from src.features.dates import to_month


def latest_hp_on_or_before(hp: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    hp = hp.copy()
    hp["month"] = to_month(hp["month"])  # ensure Timestamp
    sub = hp[hp["month"] <= month].copy()
    if sub.empty:
        sub = hp.copy()
    sub = sub.sort_values(["sa2_code","month"]).groupby("sa2_code", as_index=False).tail(1)
    keep_cols = [c for c in ["sa2_code", "median_house_price", "price_allocation_weight_sum", "price_suburb_count"] if c in sub.columns]
    return sub[keep_cols]


def _rows_for_month(df: pd.DataFrame, beta: float, k_list: list[int] | None, max_k: int | None, step: int) -> list[dict]:
    # df columns: sa2_code, month, price_pressure_prob, actual_jump, median_house_price
    df = df.dropna(subset=["median_house_price"]).copy()
    if df.empty:
        return []
    # Normalize price
    med = float(np.nanmedian(df["median_house_price"])) or 1.0
    hp_norm = df["median_house_price"].astype(float) / max(med, 1.0)
    df["score_priceaware"] = df["price_pressure_prob"].astype(float) * (hp_norm ** float(beta))

    # Baseline cumulative
    A = df.sort_values("price_pressure_prob", ascending=False).reset_index(drop=True)
    A_hits = A["actual_jump"].astype(int).to_numpy()
    A_val = (A_hits * A["median_house_price"].astype(float).to_numpy())
    A_hits_cum = np.cumsum(A_hits)
    A_val_cum = np.cumsum(A_val)

    # Price-aware cumulative
    B = df.sort_values("score_priceaware", ascending=False).reset_index(drop=True)
    B_hits = B["actual_jump"].astype(int).to_numpy()
    B_val = (B_hits * B["median_house_price"].astype(float).to_numpy())
    B_hits_cum = np.cumsum(B_hits)
    B_val_cum = np.cumsum(B_val)

    n = len(df)
    if k_list is None or len(k_list) == 0:
        k_list = list(range(1, n + 1, max(1, int(step))))
        if max_k:
            k_list = [k for k in k_list if k <= max_k]

    out = []
    m = df["month"].iloc[0]
    for K in k_list:
        if K > n:
            continue
        hitA = int(A_hits_cum[K-1]); vA = float(A_val_cum[K-1])
        hitB = int(B_hits_cum[K-1]); vB = float(B_val_cum[K-1])
        gain_abs = vB - vA
        gain_pct = (gain_abs / vA) if vA > 0 else (np.nan if vB == 0 else np.inf)
        out.append({
            "month": m, "k": int(K),
            "hits_baseline": hitA, "hits_priceaware": hitB,
            "value_hit_baseline": vA, "value_hit_priceaware": vB,
            "value_gain_abs": gain_abs, "value_gain_pct": gain_pct,
        })
    return out


def main(k_list: list[int], beta: float, start: str | None, end: str | None,
         all_k: bool, max_k: int | None, step: int) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # Predictions (prefer history)
    hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
    preds_path = hist_path if hist_path.exists() else (STAGE_DIR / "price_pressure_forecast_sa2.parquet")
    preds = pd.read_parquet(preds_path).copy()
    preds["month"] = to_month(preds["month"])
    preds["sa2_code"] = preds["sa2_code"].astype(str)

    # Realized labels
    labels = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    labels["month"] = to_month(labels["month"]) 
    labels["sa2_code"] = labels["sa2_code"].astype(str)
    labels = labels.sort_values(["sa2_code","month"]).reset_index(drop=True)
    labels["rent_prev"] = labels.groupby("sa2_code")["median_rent"].shift(1)
    labels = labels[labels["rent_prev"].notna()].copy()
    y = (labels["median_rent"] / labels["rent_prev"] - 1.0) > 0.02
    labels = labels[["sa2_code","month"]].assign(actual_jump=y.astype(int))

    # House prices (SA2)
    hp_monthly_path = STAGE_DIR / "house_prices_sa2_monthly.parquet"
    hp_snapshot_path = STAGE_DIR / "house_prices_sa2_snapshot.parquet"
    if hp_monthly_path.exists():
        hp = pd.read_parquet(hp_monthly_path).copy()
    elif hp_snapshot_path.exists():
        hp = pd.read_parquet(hp_snapshot_path).copy()
    else:
        raise SystemExit("Missing house-price parquet. Run: python -m src.cli reiwa-import-to-stage")
    rename_cols = {}
    if "allocation_weight_sum" in hp.columns:
        rename_cols["allocation_weight_sum"] = "price_allocation_weight_sum"
    if "n_suburbs" in hp.columns:
        rename_cols["n_suburbs"] = "price_suburb_count"
    if rename_cols:
        hp = hp.rename(columns=rename_cols)
    keep_cols = [c for c in ["sa2_code", "month", "median_house_price", "price_allocation_weight_sum", "price_suburb_count"] if c in hp.columns]
    hp = hp[keep_cols].copy()
    hp["sa2_code"] = hp["sa2_code"].astype(str)
    hp["month"] = to_month(hp["month"])  # normalize

    # Limit months if provided
    if start or end:
        start_m = to_month(start) if start else preds["month"].min()
        end_m = to_month(end) if end else preds["month"].max()
        preds = preds[(preds["month"] >= start_m) & (preds["month"] <= end_m)].copy()
        labels = labels[(labels["month"] >= start_m) & (labels["month"] <= end_m)].copy()

    # Join preds+labels per month, then compute value gains
    rows = []
    months = sorted(set(preds["month"]).intersection(set(labels["month"])) )
    for m in months:
        dfp = preds[preds["month"] == m].copy()
        dfl = labels[labels["month"] == m].copy()
        base = dfp.merge(dfl, on=["sa2_code","month"], how="inner")
        if base.empty:
            continue
        hp_latest = latest_hp_on_or_before(hp, m)
        df = base.merge(hp_latest, on="sa2_code", how="left")
        if all_k:
            rows.extend(_rows_for_month(df, beta, None, max_k, step))
        else:
            rows.extend(_rows_for_month(df, beta, k_list, max_k, step))

    out = pd.DataFrame(rows).sort_values(["month","k"])
    out_path = EVAL_DIR / "value_gain_from_houseprice.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure decision value gain from house prices")
    ap.add_argument("--k", type=int, nargs="+", default=[10,20])
    ap.add_argument("--beta", type=float, default=1.0, help="Exponent on price normalization in score p*(hp_norm**beta)")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--all-k", action="store_true", help="Evaluate all K from 1..N for each month (use --max-k to cap, --step to downsample)")
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()
    main(args.k, args.beta, args.start, args.end, args.all_k, args.max_k, max(1, int(args.step)))
