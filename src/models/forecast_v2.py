"""
Simplified baseline forecast that fits a scikit-learn logistic model on core
features and produces next-month rent rise probabilities.

This version intentionally skips the heavy calibration / prior shift stack so we
can inspect raw hit/miss counts easily.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import STAGE_DIR, OUTPUTS_DIR  # type: ignore[attr-defined]


# Fallback for config modules that predate OUTPUTS_DIR constant.
OUTPUTS_PATH = Path(getattr(OUTPUTS_DIR, "path", OUTPUTS_DIR)) if hasattr(OUTPUTS_DIR, "path") else Path(OUTPUTS_DIR)

# Directory to hold v2 artifacts.
V2_DIR = OUTPUTS_PATH / "v2"
V2_DIR.mkdir(parents=True, exist_ok=True)
V2_STAGE_FILE = STAGE_DIR / "price_pressure_forecast_sa2_v2.parquet"
V2_STAGE_HISTORY = STAGE_DIR / "price_pressure_forecast_sa2_history_v2.parquet"


FEATURE_COLUMNS = [
    "availability_rate",
    "availability_diff",
    "median_rent",
    "rent_change_1m",
    "rent_change_3m",
    "rent_momentum",
    "churn_rate",
    "net_stock_change",
    "stock_bonds",
    "count_lodgements",
    "count_disposals",
    "mean_days_held",
    "month_sin",
    "month_cos",
]

DEFAULT_THRESHOLDS = [0.01, 0.02, 0.03]


@dataclass
class TrainConfig:
    thresholds: list[float]
    regular_c: float = 1.0
    max_iter: int = 500


def load_base_panel() -> pd.DataFrame:
    bonds = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet")
    bonds["month"] = pd.to_datetime(bonds["month"])
    avail = pd.read_parquet(STAGE_DIR / "availability_nowcast_sa2.parquet")
    avail["month"] = pd.to_datetime(avail["month"])
    df = bonds.merge(avail, on=["sa2_code", "month"], how="left")
    df = df.sort_values(["sa2_code", "month"]).reset_index(drop=True)

    # Month dummies (cyclical)
    month = df["month"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Recent rent momentum style features.
    df["rent_change_1m"] = (
        df.groupby("sa2_code")["median_rent"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    df["rent_change_3m"] = (
        df.groupby("sa2_code")["median_rent"].pct_change(3, fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    df["availability_diff"] = df.groupby("sa2_code")["availability_rate"].diff()

    # Future target: growth between current month and next month.
    next_rent = df.groupby("sa2_code")["median_rent"].shift(-1)
    df["future_gain"] = (next_rent / df["median_rent"]) - 1.0
    df["target_month"] = df["month"] + pd.offsets.MonthBegin(1)

    # Drop rows without available features or targets.
    df = df[df["median_rent"].notna()].copy()
    return df


def build_model(c_value: float, max_iter: int) -> Pipeline:
    clf = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("scaler", StandardScaler()), ("logit", clf)])


def prepare_train_eval_frames(df: pd.DataFrame, month_cutoffs: Iterable[pd.Timestamp]) -> dict[pd.Timestamp, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    For each cutoff month, return (train, eval) where train contains data up to the previous month,
    and eval contains the specified month.
    """
    grouped: dict[pd.Timestamp, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for eval_month in month_cutoffs:
        train_mask = df["month"] < eval_month
        eval_mask = df["month"] == eval_month
        train_df = df[train_mask & df["future_gain"].notna()].copy()
        eval_df = df[eval_mask].copy()
        grouped[eval_month] = (train_df, eval_df)
    return grouped


def fit_and_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: TrainConfig) -> dict[str, pd.Series]:
    model = build_model(config.regular_c, config.max_iter)
    X_train = train_df[FEATURE_COLUMNS].fillna(0.0)
    y_train = (train_df["future_gain"] > config.thresholds[1]).astype(int)  # use 2% target for fitting
    model.fit(X_train, y_train)

    probs = model.predict_proba(eval_df[FEATURE_COLUMNS].fillna(0.0))[:, 1]
    return {
        "prob": pd.Series(probs, index=eval_df.index),
    }


def evaluate_month(eval_df: pd.DataFrame, probs: pd.Series, thresholds: list[float]) -> pd.DataFrame:
    results = []
    for thr in thresholds:
        actual = (eval_df["future_gain"] > thr).astype(int)
        preds = (probs >= 0.5).astype(int)
        tp = int(((preds == 1) & (actual == 1)).sum())
        fp = int(((preds == 1) & (actual == 0)).sum())
        tn = int(((preds == 0) & (actual == 0)).sum())
        fn = int(((preds == 0) & (actual == 1)).sum())
        total = tp + fp + tn + fn
        results.append({
            "month": eval_df["target_month"].iloc[0].strftime("%Y-%m"),
            "threshold": thr,
            "actual_rises": int(actual.sum()),
            "flagged": int((preds == 1).sum()),
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "accuracy": (tp + tn) / total if total else 0.0,
        })
    return pd.DataFrame(results)


def main(thresholds: list[float], final_month: str) -> None:
    df = load_base_panel()
    df["month"] = pd.to_datetime(df["month"])

    config = TrainConfig(thresholds=thresholds)

    eval_months = sorted(df.loc[df["future_gain"].notna(), "month"].unique())
    # Keep recent months (exclude early warm-up).
    eval_months = [m for m in eval_months if m >= pd.Timestamp("2024-03-01")]
    month_map = prepare_train_eval_frames(df, eval_months)

    all_eval_rows = []
    history_rows = []

    for month, (train_df, eval_df) in month_map.items():
        if train_df.empty or eval_df.empty:
            continue
        preds = fit_and_predict(train_df, eval_df, config)
        eval_metrics = evaluate_month(eval_df, preds["prob"], thresholds)
        all_eval_rows.append(eval_metrics)
        eval_df = eval_df.copy()
        eval_df["forecast_prob"] = preds["prob"]
        eval_df["forecast_month"] = eval_df["target_month"]
        history_rows.append(eval_df[["sa2_code", "forecast_month", "forecast_prob"]])
        out_path = V2_DIR / f"predictions_{eval_df['target_month'].iloc[0].strftime('%Y-%m')}.parquet"
        eval_df[["sa2_code", "forecast_month", "forecast_prob"]].rename(columns={"forecast_month": "month"}).to_parquet(out_path, index=False)

    if all_eval_rows:
        summary = pd.concat(all_eval_rows, ignore_index=True)
        summary.to_csv(V2_DIR / "evaluation_summary.csv", index=False)

    # Produce latest forecast (for month without future gain).
    target_month = pd.Timestamp(final_month)
    df_latest = df[df["month"] == target_month].copy()
    train_latest = df[(df["month"] < target_month) & df["future_gain"].notna()].copy()
    latest_pred = fit_and_predict(train_latest, df_latest, config)
    df_latest = df_latest.copy()
    df_latest["forecast_prob"] = latest_pred["prob"]
    df_latest["forecast_month"] = df_latest["target_month"]
    df_latest[["sa2_code", "forecast_month", "forecast_prob"]].rename(columns={"forecast_month": "month"}).to_parquet(V2_DIR / f"forecast_{(target_month + pd.offsets.MonthBegin(1)).strftime('%Y-%m')}.parquet", index=False)

    combined_history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame(columns=["sa2_code", "forecast_month", "forecast_prob"])
    if not combined_history.empty:
        combined_history = combined_history.rename(columns={"forecast_month": "month", "forecast_prob": "price_pressure_prob"})
        combined_history["rent_jump_threshold"] = config.thresholds[1] if len(config.thresholds) > 1 else 0.02
        combined_history.to_parquet(V2_STAGE_HISTORY, index=False)

    latest_out = df_latest[["sa2_code", "forecast_month", "forecast_prob"]].rename(
        columns={"forecast_month": "month", "forecast_prob": "price_pressure_prob"}
    )
    latest_out["rent_jump_threshold"] = config.thresholds[1] if len(config.thresholds) > 1 else 0.02
    latest_out.to_parquet(V2_STAGE_FILE, index=False)

    pd.options.display.max_rows = 10
    print("Saved evaluation summary to", V2_DIR / "evaluation_summary.csv")
    print("Latest forecast saved to", V2_DIR / f"forecast_{(target_month + pd.offsets.MonthBegin(1)).strftime('%Y-%m')}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simplified logistic forecast (v2 baseline).")
    ap.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS,
                    help="Rent growth thresholds (default 0.01 0.02 0.03)")
    ap.add_argument("--latest-month", type=str, default="2025-09",
                    help="Month to generate the current forecast for (format YYYY-MM)")
    args = ap.parse_args()
    main(args.thresholds, args.latest_month)
