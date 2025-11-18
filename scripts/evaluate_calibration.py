"""
Evaluate calibration and alert tiers for a given forecast month.

Usage:
    python -m scripts.evaluate_calibration --month 2025-09 \
        --jump-threshold 0.02 --out outputs/tables/calibration_summary_2025-09.json

The script:
  * Loads historical forecast probabilities (prefers the thresholded history file).
  * Joins realized rental jumps from `data_stage/bonds_panel_sa2.parquet`.
  * Fits a logistic calibration layer on all months strictly before the target month.
  * Reports baseline vs calibrated metrics (AUC, AP, Brier, mean prob, top-N precision/recall).
  * Emits tiered alert suggestions (critical/high/medium) from calibrated probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import STAGE_DIR


def _load_forecasts() -> pd.DataFrame:
    """Return forecast history, falling back to the generic file if needed."""
    preferred = STAGE_DIR / "price_pressure_forecast_sa2_history_thr0p020.parquet"
    fallback = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"

    frames: list[pd.DataFrame] = []
    if preferred.exists():
        df_pref = pd.read_parquet(preferred)
        df_pref["source_path"] = str(preferred)
        frames.append(df_pref)
        if df_pref["month"].nunique() >= 2:
            return df_pref

    if fallback.exists():
        df_fb = pd.read_parquet(fallback)
        df_fb["source_path"] = str(fallback)
        frames.append(df_fb)

    if not frames:
        raise SystemExit("No forecast history parquet found under data_stage/.")

    # Combine preferred + fallback (drops duplicates, keeps preferred data when overlapping).
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["sa2_code", "month"], keep="first")
    combined["source_path"] = ",".join({str(path) for path in [preferred, fallback] if path.exists()})
    return combined


def _load_actuals(threshold: float) -> pd.DataFrame:
    """Construct realized jump labels from the SA2 bonds panel."""
    bonds_path = STAGE_DIR / "bonds_panel_sa2.parquet"
    if not bonds_path.exists():
        raise SystemExit(f"Missing realized rents parquet: {bonds_path}")

    df = pd.read_parquet(bonds_path).copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["sa2_code", "month"])
    df["rent_prev"] = df.groupby("sa2_code")["median_rent"].shift(1)
    df["pct_change"] = df["median_rent"] / df["rent_prev"] - 1.0
    df["actual_jump"] = (df["pct_change"] > threshold).astype(int)
    return df


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_prob": float(np.mean(p)),
    }


def _topn_stats(y: np.ndarray, p: np.ndarray, n_values: Iterable[int]) -> list[dict[str, float]]:
    order = np.argsort(-p)  # descending
    y_ord = y[order]
    p_ord = p[order]
    total_hits = float(y.sum()) or 1.0
    stats = []
    for n in n_values:
        if n > len(y_ord):
            break
        hits = float(y_ord[:n].sum())
        precision = hits / n
        recall = hits / total_hits
        stats.append(
            {
                "N": int(n),
                "precision": float(precision),
                "recall": float(recall),
                "cutoff": float(p_ord[n - 1]),
            }
        )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate calibration and alert tiers for a forecast month.")
    ap.add_argument("--month", type=str, default=None, help="Month to evaluate (YYYY-MM). Defaults to latest overlap.")
    ap.add_argument("--jump-threshold", type=float, default=0.02, help="Rent jump threshold (default 0.02).")
    ap.add_argument("--out", type=str, required=True, help="Path to JSON output file.")
    ap.add_argument("--tiers", type=str, nargs=3, default=["10", "20", "30"],
                    help="Number of SA2s in critical/high/medium tiers (default 10/20/30).")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    forecasts = _load_forecasts()
    forecasts["month"] = pd.to_datetime(forecasts["month"])
    actuals = _load_actuals(args.jump_threshold)

    joined = forecasts.merge(
        actuals[["sa2_code", "month", "actual_jump"]],
        on=["sa2_code", "month"],
        how="left",
    ).dropna(subset=["actual_jump"])
    joined["actual_jump"] = joined["actual_jump"].astype(int)

    if joined.empty:
        raise SystemExit("No overlapping forecasts and actuals found; check data staging.")

    months = sorted(joined["month"].unique())
    month_lookup = {m.strftime("%Y-%m"): m for m in months}
    if args.month:
        target_month = month_lookup.get(args.month)
        if target_month is None:
            raise SystemExit(f"Month {args.month} not available; choices: {sorted(month_lookup)}")
    else:
        target_month = months[-1]

    train_mask = joined["month"] < target_month
    train = joined.loc[train_mask].copy()
    latest = joined.loc[joined["month"] == target_month].copy()

    if train["actual_jump"].nunique() < 2:
        raise SystemExit("Not enough variation in training months to calibrate (need at least one jump and non-jump).")

    model = LogisticRegression()
    model.fit(train[["price_pressure_prob"]], train["actual_jump"])
    joined["prob_calibrated_lr"] = model.predict_proba(joined[["price_pressure_prob"]])[:, 1]
    latest["prob_calibrated_lr"] = model.predict_proba(latest[["price_pressure_prob"]])[:, 1]

    y_latest = latest["actual_jump"].to_numpy(dtype=int)
    base_probs = latest["price_pressure_prob"].to_numpy(dtype=float)
    cal_probs = latest["prob_calibrated_lr"].to_numpy(dtype=float)

    metrics = {
        "baseline": _metrics(y_latest, base_probs),
        "logit": _metrics(y_latest, cal_probs),
    }
    n_values = [10, 20, 40, 60, 80, 100]
    metrics["baseline"]["topN"] = _topn_stats(y_latest, base_probs, n_values)
    metrics["logit"]["topN"] = _topn_stats(y_latest, cal_probs, n_values)

    # Build tiered alert suggestion from calibrated probabilities
    tier_sizes = [int(x) for x in args.tiers]
    order = latest.sort_values("prob_calibrated_lr", ascending=False).reset_index(drop=True)
    total_hits = float(y_latest.sum()) or 1.0
    tier_labels = ["critical", "high", "medium"]
    tier_summary: dict[str, dict[str, float]] = {}
    start = 0
    tier_map: dict[str, str] = {}
    for label, size in zip(tier_labels, tier_sizes):
        end = start + size
        subset = order.iloc[start:end].copy()
        hits = float(subset["actual_jump"].sum())
        for code in subset["sa2_code"].astype(str):
            tier_map[code] = label
        tier_summary[label] = {
            "count": int(len(subset)),
            "hits": int(hits),
            "precision": float(hits / len(subset)) if len(subset) else 0.0,
            "recall": float(hits / total_hits),
            "min_prob": float(subset["prob_calibrated_lr"].min()) if len(subset) else 0.0,
            "max_prob": float(subset["prob_calibrated_lr"].max()) if len(subset) else 0.0,
        }
        start = end
    latest["tier"] = latest["sa2_code"].astype(str).map(tier_map).fillna("none")

    summary = {
        "month": target_month.strftime("%Y-%m"),
        "forecast_source": forecasts["source_path"].iloc[0] if "source_path" in forecasts.columns else None,
        "jump_threshold": args.jump_threshold,
        "total_sa2": int(len(latest)),
        "actual_jumps": int(y_latest.sum()),
        "model": {
            "coef": float(model.coef_.ravel()[0]),
            "intercept": float(model.intercept_.ravel()[0]),
        },
        "metrics": metrics,
        "tier_summary": tier_summary,
    }

    # Also store the per-SA2 calibrated probabilities for the target month.
    probs_path = out_path.with_name(out_path.stem + "_probabilities.csv")
    cols = ["sa2_code", "month", "price_pressure_prob", "prob_calibrated_lr", "actual_jump", "tier"]
    latest.loc[:, cols].to_csv(probs_path, index=False)

    tier_paths: dict[str, str] = {}
    for label in tier_labels:
        tier_df = latest[latest["tier"] == label].copy()
        if tier_df.empty:
            continue
        tier_df = tier_df.sort_values("prob_calibrated_lr", ascending=False)
        tier_path = out_path.with_name(f"{out_path.stem}_{label}.csv")
        tier_df.to_csv(tier_path, index=False)
        tier_paths[label] = str(tier_path)
    summary["tier_files"] = tier_paths

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[info] Wrote summary JSON to {out_path}")
    print(f"[info] Wrote per-SA2 probabilities to {probs_path}")
    for label, path in tier_paths.items():
        print(f"[info] Wrote {label} tier table to {path}")


if __name__ == "__main__":
    main()
