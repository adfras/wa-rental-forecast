"""Run the forecast pipeline for multiple rent-rise thresholds and report accuracy.

Usage (from repo root, venv activated):

    python -m tools.run_multi_threshold --thresholds 0.01 0.02 0.03 \
        --draws 800 --tune 800 --chains 2 --cores 2 \
        --optimize-by accuracy

This will:
  * Fit the Bayesian forecast for each threshold (with suffix per threshold)
  * Evaluate each run, capturing monthly accuracy/precision/recall
  * Save evaluation artefacts under outputs/evaluations/multi_threshold/<suffix>
  * Print a summary table and copy the best-performing history back to the
    default location for downstream tooling (map, reports).

The script is meant for pilot/ops comparisons, so thresholds, sampling depth and
metrics are configurable.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.config import STAGE_DIR, OUTPUTS_DIR
from src.models.forecast import fit_forecast
from src.reporting.evaluate_forecasts import main as eval_forecasts


@dataclass
class ThresholdResult:
    rent_threshold: float
    suffix: str
    latest_month: str | None
    accuracy_at_thr: float | None
    precision_at_thr: float | None
    recall_at_thr: float | None
    f1_best: float | None
    prob_threshold: float | None
    brier: float | None
    auc: float | None


def fmt_threshold_suffix(thr: float) -> str:
    """Generate a filesystem-friendly suffix like 'thr0p020'."""
    sign = "m" if thr < 0 else ""
    value = abs(thr)
    formatted = f"{value:.3f}".replace(".", "p")
    return f"thr{sign}{formatted}"


def copy_history_for_evaluation(suffix: str) -> Path:
    src = STAGE_DIR / f"price_pressure_forecast_sa2_history_{suffix}.parquet"
    dst = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Forecast history not found for suffix {suffix}: {src}")
    shutil.copy2(src, dst)
    return dst


def collect_summary(rent_threshold: float, suffix: str, target_dir: Path) -> ThresholdResult:
    summary_path = OUTPUTS_DIR / "evaluations" / "forecast_eval_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Evaluation summary not produced; expected "
                                f"{summary_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    dst_summary = target_dir / f"forecast_eval_summary_{suffix}.csv"
    shutil.copy2(summary_path, dst_summary)

    # Move per-month detail and calibration files to keep directories tidy
    eval_dir = OUTPUTS_DIR / "evaluations"
    for pattern in ["forecast_eval_details_*.csv", "forecast_calibration_*.csv",
                    "forecast_calibration_clusters.csv"]:
        for path in eval_dir.glob(pattern):
            shutil.move(path, target_dir / path.name)

    df = pd.read_csv(dst_summary)
    if df.empty:
        return ThresholdResult(
            rent_threshold=rent_threshold,
            suffix=suffix,
            latest_month=None,
            accuracy_at_thr=None,
            precision_at_thr=None,
            recall_at_thr=None,
            f1_best=None,
            prob_threshold=None,
            brier=None,
            auc=None,
        )

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")
    latest = df.iloc[-1]
    return ThresholdResult(
        rent_threshold=rent_threshold,
        suffix=suffix,
        latest_month=latest["month"].strftime("%Y-%m"),
        accuracy_at_thr=_maybe_float(latest.get("accuracy_at_thr")),
        precision_at_thr=_maybe_float(latest.get("precision_at_thr")),
        recall_at_thr=_maybe_float(latest.get("recall_at_thr")),
        f1_best=_maybe_float(latest.get("best_f1")),
        prob_threshold=_maybe_float(latest.get("threshold")),
        brier=_maybe_float(latest.get("brier")),
        auc=_maybe_float(latest.get("auc")),
    )


def _maybe_float(value) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def thr_from_suffix(suffix: str) -> float:
    core = suffix.replace("thr", "")
    sign = -1 if core.startswith("m") else 1
    if sign == -1:
        core = core[1:]
    value = core.replace("p", ".")
    try:
        return sign * float(value)
    except ValueError:
        return math.nan


def choose_best(results: Iterable[ThresholdResult], metric: str) -> ThresholdResult:
    metric_map = {
        "accuracy": lambda r: r.accuracy_at_thr,
        "precision": lambda r: r.precision_at_thr,
        "recall": lambda r: r.recall_at_thr,
        "f1": lambda r: r.f1_best,
        "auc": lambda r: r.auc,
    }
    scorer = metric_map[metric]
    best = None
    best_value = -math.inf
    for res in results:
        value = scorer(res)
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best = res
        elif value == best_value and best is not None:
            if (res.rent_threshold or math.inf) < (best.rent_threshold or math.inf):
                best = res
    if best is None:
        print("Warning: no actual evaluation metrics available; defaulting to first threshold.")
        return next(iter(results))
    return best


def run_threshold(thr: float, args) -> ThresholdResult:
    suffix = fmt_threshold_suffix(thr)
    start = time.time()
    print(f"\n=== Threshold {thr:.3%} ({suffix}) ===")
    history_path = STAGE_DIR / f"price_pressure_forecast_sa2_history_{suffix}.parquet"
    if args.draws <= 0:
        if args.force:
            print("draws<=0 requested; ignoring --force because no sampling will run.")
        if not history_path.exists():
            raise ValueError(
                "No existing history available to reuse for "
                f"suffix {suffix}; run with positive --draws at least once."
            )
        print("Skipping fit because draws<=0; reusing saved history for evaluation.")
    elif history_path.exists() and not args.force:
        print("History already exists; skipping fit (use --force to refit).")
    else:
        try:
            fit_forecast(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                recency_half_life=args.recency_half_life,
                bias_correct_l6=not args.no_bias_correct_l6,
                calibrate_isotonic=not args.no_calibrate_isotonic,
                prior_shift=args.prior_shift,
                auto_calibrate=not args.no_auto_calibrate,
                target_accept=args.target_accept,
                sampler=args.sampler,
                init=args.init,
                max_treedepth=args.max_treedepth,
                threshold=thr,
                output_suffix=suffix,
            )
        except ValueError as exc:
            if "Not enough samples" in str(exc):
                print("Sampling aborted (not enough samples). Try increasing draws/tune or using --force after adjusting parameters.")
                raise
            raise
    copy_history_for_evaluation(suffix)
    print("Evaluating…")
    eval_forecasts(threshold=thr)

    target_dir = OUTPUTS_DIR / "evaluations" / "multi_threshold" / suffix
    result = collect_summary(thr, suffix, target_dir)
    duration = time.time() - start
    print(f"Completed threshold {thr:.3%} in {duration/60:.1f} min")
    return result


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run forecast pipeline for multiple thresholds and select the best performer.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.03],
                        help="List of rent-change thresholds to test (fractional, e.g., 0.02 for 2%%).")
    parser.add_argument("--draws", type=int, default=800,
                        help="Posterior draws per chain (set 0 to skip fitting and reuse existing history).")
    parser.add_argument("--tune", type=int, default=800)
    parser.add_argument("--chains", type=int, default=4,
                        help="Number of MCMC chains (default 4 to match production).")
    parser.add_argument("--cores", type=int, default=4,
                        help="Worker cores for PyMC sampling (default 4).")
    parser.add_argument("--recency-half-life", type=float, default=None,
                        help="Half-life in months for sample weighting (default: config value).")
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--sampler", choices=["pymc", "blackjax", "numpyro"], default="pymc")
    parser.add_argument("--init", default="jitter+adapt_diag")
    parser.add_argument("--max-treedepth", type=int, default=15)
    parser.add_argument("--no-bias-correct-l6", action="store_true")
    parser.add_argument("--no-calibrate-isotonic", action="store_true")
    parser.add_argument("--prior-shift", action="store_true")
    parser.add_argument("--no-auto-calibrate", action="store_true")
    parser.add_argument("--optimize-by", choices=["accuracy", "precision", "recall", "f1", "auc"], default="accuracy",
                        help="Metric used to choose the best-performing threshold (from latest month).")
    parser.add_argument("--restore-best-history", action="store_true",
                        help="After running all thresholds, copy the best history back to the default path.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-fitting even if history files already exist.")
    args = parser.parse_args(argv)

    if args.draws < 0:
        parser.error("--draws must be non-negative.")
    if args.tune < 0:
        parser.error("--tune must be non-negative.")
    if args.chains <= 0:
        parser.error("--chains must be at least 1.")
    if args.cores <= 0:
        parser.error("--cores must be at least 1.")

    thresholds = sorted(set(args.thresholds))
    output_root = OUTPUTS_DIR / "evaluations" / "multi_threshold"
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[ThresholdResult] = []
    for thr in thresholds:
        result = run_threshold(thr, args)
        results.append(result)

    summary_df = pd.DataFrame([
        {
            "rent_threshold": res.rent_threshold,
            "suffix": res.suffix,
            "latest_month": res.latest_month,
            "accuracy_at_thr": res.accuracy_at_thr,
            "precision_at_thr": res.precision_at_thr,
            "recall_at_thr": res.recall_at_thr,
            "best_f1": res.f1_best,
            "prob_threshold": res.prob_threshold,
            "auc": res.auc,
            "brier": res.brier,
        }
        for res in results
    ])
    summary_path = output_root / "multi_threshold_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best = choose_best(results, args.optimize_by)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
    print(f"\nBest threshold ({args.optimize_by}): {best.rent_threshold:.3%} (suffix {best.suffix})")

    if args.restore_best_history:
        src = STAGE_DIR / f"price_pressure_forecast_sa2_history_{best.suffix}.parquet"
        dst = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Restored best history to {dst}")
        else:
            print(f"Warning: best history not found at {src}")

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
