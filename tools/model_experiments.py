"""
Run comparative experiments across multiple classifiers / feature sets and
log simple accuracy tables for rent-rise forecasts.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import STAGE_DIR, OUTPUTS_DIR  # type: ignore[attr-defined]

OUTPUTS_PATH = Path(getattr(OUTPUTS_DIR, "path", OUTPUTS_DIR)) if hasattr(OUTPUTS_DIR, "path") else Path(OUTPUTS_DIR)
EXP_DIR = OUTPUTS_PATH / "v2"
EXP_DIR.mkdir(parents=True, exist_ok=True)


BASE_FEATURES = [
    "availability_rate",
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

EXTRA_FEATURES = [
    "availability_diff",
    "avail_x_churn",
    "rent_change_1m_x_avail",
    "rent_change_3m_x_churn",
    "gt_rent_intent",
    "gt_vacancy_intent",
    "wa_unemp_rate_sa",
    "wa_build_approvals_num",
    "perth_cpi",
    "wa_bonds_rent_index",
]

THRESHOLDS = [0.01, 0.02, 0.03]


def load_panel() -> pd.DataFrame:
    bonds = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet")
    bonds["month"] = pd.to_datetime(bonds["month"])
    avail = pd.read_parquet(STAGE_DIR / "availability_nowcast_sa2.parquet")
    avail["month"] = pd.to_datetime(avail["month"])
    df = bonds.merge(avail, on=["sa2_code", "month"], how="left")
    df = df.sort_values(["sa2_code", "month"]).reset_index(drop=True)

    month = df["month"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["rent_change_1m"] = (
        df.groupby("sa2_code")["median_rent"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    df["rent_change_3m"] = (
        df.groupby("sa2_code")["median_rent"].pct_change(3, fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    df["availability_diff"] = df.groupby("sa2_code")["availability_rate"].diff()

    next_rent = df.groupby("sa2_code")["median_rent"].shift(-1)
    df["future_gain"] = (next_rent / df["median_rent"]) - 1.0
    df["target_month"] = df["month"] + pd.offsets.MonthBegin(1)

    # External signals (state-wide) merged by month.
    ext_path = STAGE_DIR / "external_signals.parquet"
    if ext_path.exists():
        ext = pd.read_parquet(ext_path)
        ext["month"] = pd.to_datetime(ext["month"])
        df = df.merge(ext, on="month", how="left")

    # House price growth (if available) as extra signal.
    hp_path = STAGE_DIR / "house_prices_sa2_monthly.parquet"
    if hp_path.exists():
        hp = pd.read_parquet(hp_path)
        hp["month"] = pd.to_datetime(hp["month"])
        hp = hp.rename(columns={"median_house_price": "hp_median_price"})
        df = df.merge(hp[["sa2_code", "month", "hp_median_price"]], on=["sa2_code", "month"], how="left")
        df["hp_change_3m"] = (
            df.groupby("sa2_code")["hp_median_price"].pct_change(3, fill_method=None).replace([np.inf, -np.inf], np.nan)
        )
    df["avail_x_churn"] = df["availability_rate"] * df["churn_rate"]
    df["rent_change_1m_x_avail"] = df["rent_change_1m"] * df["availability_rate"]
    df["rent_change_3m_x_churn"] = df["rent_change_3m"] * df["churn_rate"]
    return df


def make_datasets(df: pd.DataFrame, min_month: str = "2024-03") -> Dict[str, pd.Series]:
    mask = df["target_month"] >= pd.Timestamp(min_month)
    df = df.loc[mask & df["future_gain"].notna()].copy()
    return df


def train_eval_split(df: pd.DataFrame) -> Dict[pd.Timestamp, Tuple[pd.DataFrame, pd.DataFrame]]:
    groups: Dict[pd.Timestamp, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    unique_months = sorted(df["target_month"].unique())
    for month in unique_months:
        train_mask = df["target_month"] < month
        eval_mask = df["target_month"] == month
        train_df = df[train_mask].copy()
        eval_df = df[eval_mask].copy()
        if not train_df.empty and not eval_df.empty:
            groups[month] = (train_df, eval_df)
    return groups


def build_models() -> Dict[str, Tuple[List[str], Pipeline]]:
    models: Dict[str, Tuple[List[str], Pipeline]] = {}
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)),
    ])
    models["logistic_baseline"] = (BASE_FEATURES, logit)

    logit_ext = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)),
    ])
    models["logistic_extended"] = (BASE_FEATURES + EXTRA_FEATURES, logit_ext)

    hgb = HistGradientBoostingClassifier(
        max_depth=None,
        max_iter=400,
        learning_rate=0.05,
        min_samples_leaf=25,
        l2_regularization=0.0,
    )
    models["hist_gradient_boosting"] = (BASE_FEATURES + EXTRA_FEATURES, hgb)
    return models


def evaluate_model(name: str,
                   feature_cols: List[str],
                   model,
                   splits: Dict[pd.Timestamp, Tuple[pd.DataFrame, pd.DataFrame]],
                   thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for target_month, (train_df, eval_df) in splits.items():
        X_train = train_df[feature_cols].fillna(0.0)
        y_train = (train_df["future_gain"] > thresholds[1]).astype(int)
        model.fit(X_train, y_train)
        probs = model.predict_proba(eval_df[feature_cols].fillna(0.0))[:, 1]
        eval_df = eval_df.copy()
        eval_df["pred_prob"] = probs
        for thr in thresholds:
            actual = (eval_df["future_gain"] > thr).astype(int)
            preds50 = (eval_df["pred_prob"] >= 0.5).astype(int)
            preds10 = (eval_df["pred_prob"] >= 0.1).astype(int)
            for label, preds in [("cut_0.50", preds50), ("cut_0.10", preds10)]:
                tp = int(((preds == 1) & (actual == 1)).sum())
                fp = int(((preds == 1) & (actual == 0)).sum())
                tn = int(((preds == 0) & (actual == 0)).sum())
                fn = int(((preds == 0) & (actual == 1)).sum())
                total = tp + fp + tn + fn
                rows.append({
                    "model": name,
                    "target_month": target_month.strftime("%Y-%m"),
                    "threshold": thr,
                    "decision_cut": label,
                    "flagged": int((preds == 1).sum()),
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "accuracy": (tp + tn) / total if total else 0.0,
                })
    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model", "threshold", "decision_cut"])
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )
    return summary


def compare_to_baseline(summary: pd.DataFrame, baseline: str = "logistic_baseline") -> pd.DataFrame:
    base = summary[summary["model"] == baseline].copy()
    merged = summary.merge(
        base,
        on=["threshold", "decision_cut"],
        suffixes=("", "_baseline"),
        how="left",
    )
    for metric in ["mean_precision", "mean_recall", "mean_accuracy"]:
        merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_baseline"]
    return merged


def main():
    df = load_panel()
    df = make_datasets(df)
    splits = train_eval_split(df)
    models = build_models()
    all_results = []
    for name, (features, model) in models.items():
        res = evaluate_model(name, features, model, splits, THRESHOLDS)
        all_results.append(res)
    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(EXP_DIR / "experiments_detailed.csv", index=False)
    summary = aggregate_results(full)
    summary.to_csv(EXP_DIR / "experiments_summary.csv", index=False)
    comp = compare_to_baseline(summary)
    comp.to_csv(EXP_DIR / "experiments_vs_baseline.csv", index=False)
    print("Saved detailed results:", EXP_DIR / "experiments_detailed.csv")
    print("Saved summary:", EXP_DIR / "experiments_summary.csv")
    print("Saved comparison:", EXP_DIR / "experiments_vs_baseline.csv")


if __name__ == "__main__":
    main()
