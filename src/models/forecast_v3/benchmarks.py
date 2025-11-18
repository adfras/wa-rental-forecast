"""Baseline benchmarking utilities for forecast v3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # prefer tqdm if available for progress bars
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .config import CONFIG, PATHS
from .data import prepare_dataset

try:  # optional dependencies
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

if lgb is not None and hasattr(lgb, "set_config"):
    lgb.set_config(verbosity=-1)

try:  # optional dependency
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        avg_prob = float(y_prob[mask].mean())
        avg_true = float(y_true[mask].mean())
        total += mask.sum() / y_true.size * abs(avg_prob - avg_true)
    return total


FEATURE_CORE = [
    "median_rent",
    "availability_rate",
    "rent_change_1m",
    "rent_change_3m",
    "churn_rate",
    "net_stock_change",
    "stock_bonds",
    "count_lodgements",
    "count_disposals",
    "mean_days_held",
    "month_sin",
    "month_cos",
]

FEATURE_EXTENDED = FEATURE_CORE + [
    "availability_diff",
    "stock_change_pct",
    "avail_x_churn",
    "rent_change_1m_x_avail",
    "rent_change_lag_3",
    "availability_rate_lag_3",
    "churn_rate_lag_3",
    "rent_change_lag_6",
    "availability_rate_lag_6",
    "airbnb_density",
    "airbnb_revpar",
    "airbnb_density_lag_1",
    "airbnb_density_lag_3",
    "airbnb_density_lag_6",
    "macro_unemployment_rate",
    "macro_policy_rate",
    "macro_nominal_gdp",
    "macro_disposable_income",
    "macro_vacancy_rate",
    "macro_inflation_expectations",
    "macro_house_price_index",
    "macro_unemployment_rate_diff_1m",
    "macro_unemployment_rate_yoy",
]

MULTIMODAL_PREFIXES = ("text_", "img_")


@dataclass
class SplitConfig:
    eval_months: List[pd.Timestamp]
    thresholds: Tuple[float, ...]


def build_splits(df: pd.DataFrame, *, thresholds: Tuple[float, ...]) -> SplitConfig:
    eval_months = sorted(df["target_month"].dropna().unique())
    eval_months = [m for m in eval_months if m >= pd.Timestamp(CONFIG.eval_start_month)]
    return SplitConfig(eval_months=eval_months, thresholds=thresholds)


def baseline_models() -> Dict[str, Tuple[List[str], object]]:
    models: Dict[str, Tuple[List[str], object]] = {}
    logit_core = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
        )),
    ])
    logit_ext = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
        )),
    ])
    hgb = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=20,
    )
    models["logistic_core"] = (FEATURE_CORE, logit_core)
    models["logistic_extended"] = (FEATURE_EXTENDED, logit_ext)
    models["hgb_extended"] = (FEATURE_EXTENDED, hgb)

    if lgb is not None:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            verbosity=-1,
        )
        models["lightgbm_extended"] = (FEATURE_EXTENDED, lgb_clf)

    if CatBoostClassifier is not None:
        cat_clf = CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1.0, 2.0],
        )
        models["catboost_extended"] = (FEATURE_EXTENDED, cat_clf)

    base_estimators = []
    if lgb is not None:
        base_estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
        )))
    base_estimators.append(("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )))
    base_estimators.append(("logit", Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
        )),
    ])))
    if base_estimators:
        stack_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=RidgeClassifier(alpha=1.0),
            passthrough=True,
            n_jobs=-1,
        )
        models["stacked_ensemble"] = (FEATURE_EXTENDED, stack_clf)

    return models


def evaluate_model(
    name: str,
    features: List[str],
    model,
    df: pd.DataFrame,
    split_cfg: SplitConfig,
    decision_cuts: Iterable[float],
    *,
    label_column: str,
    growth_threshold: float,
    severity_column: str,
    topk: int = 20,
) -> pd.DataFrame:
    rows = []
    month_iter: Iterable[pd.Timestamp]
    if tqdm is not None:
        month_iter = tqdm(
            split_cfg.eval_months,
            desc=f"{name} months",
            unit="month",
            leave=False,
        )
    else:
        month_iter = split_cfg.eval_months

    for month in month_iter:
        train = df[(df["target_month"] < month) & df[label_column].notna()].copy()
        test = df[df["target_month"] == month].copy()
        if train.empty or test.empty:
            continue
        test = test[test[label_column].notna()].copy()
        if test.empty:
            continue
        available_features = [feat for feat in features if feat in train.columns]
        # Include multimodal embeddings if available
        extra_feats = [col for col in train.columns if col.startswith(MULTIMODAL_PREFIXES)]
        available_features += [col for col in extra_feats if col not in available_features]
        available_features = list(dict.fromkeys(available_features))
        if not available_features:
            continue
        X_train = train[available_features].fillna(0.0)
        y_train = (train[label_column] > growth_threshold).astype(int)
        model.fit(X_train, y_train)
        X_test = test[available_features].fillna(0.0)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            preds = model.predict(X_test)
            probs = preds.astype(float)
        severity_available = severity_column in train.columns and severity_column in test.columns
        severity_pred = np.zeros(len(test), dtype=float)
        severity_actual_vals = np.zeros(len(test), dtype=float)
        if severity_available:
            train_severity = train[(train[label_column] > growth_threshold) & train[severity_column].notna()].copy()
            if len(train_severity) >= 20:
                X_reg = train_severity[available_features].fillna(0.0)
                y_reg = train_severity[severity_column].astype(float)
                reg = HistGradientBoostingRegressor(max_depth=4, max_iter=400, learning_rate=0.05)
                try:
                    reg.fit(X_reg, y_reg)
                    severity_pred = np.clip(reg.predict(X_test), 0.0, None)
                except Exception:
                    severity_pred = np.zeros(len(test), dtype=float)
            severity_actual_vals = test[severity_column].astype(float).fillna(0.0).to_numpy()

        expected_gain = probs * severity_pred
        actual_binary = (test[label_column] > growth_threshold).astype(int).to_numpy()
        probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        if len(np.unique(actual_binary)) > 1:
            logloss = float(log_loss(actual_binary, probs_clipped, labels=[0, 1]))
            roc_auc = float(roc_auc_score(actual_binary, probs))
            avg_prec = float(average_precision_score(actual_binary, probs))
        else:
            logloss = float("nan")
            roc_auc = float("nan")
            avg_prec = float("nan")
        try:
            brier = float(brier_score_loss(actual_binary, probs_clipped))
        except Exception:
            brier = float("nan")
        ece = _expected_calibration_error(actual_binary, probs)

        mask_pos = actual_binary == 1
        gain_mae = (
            float(mean_absolute_error(severity_actual_vals[mask_pos], severity_pred[mask_pos]))
            if np.any(mask_pos)
            else float("nan")
        )
        gain_rmse = (
            float(np.sqrt(mean_squared_error(severity_actual_vals[mask_pos], severity_pred[mask_pos])))
            if np.any(mask_pos)
            else float("nan")
        )

        k = max(1, min(topk, len(expected_gain)))
        idx_top = np.argsort(-expected_gain)[:k]
        top_probs = probs_clipped[idx_top]
        top_actual = actual_binary[idx_top]
        if len(np.unique(top_actual)) > 1:
            topk_log_loss = float(log_loss(top_actual, top_probs, labels=[0, 1]))
        else:
            topk_log_loss = float("nan")
        try:
            topk_brier = float(brier_score_loss(top_actual, top_probs))
        except Exception:
            topk_brier = float("nan")
        topk_actual_gain = float(severity_actual_vals[idx_top].sum())
        topk_expected_gain = float(expected_gain[idx_top].sum())
        topk_gain_diff = topk_expected_gain - topk_actual_gain

        for thr in split_cfg.thresholds:
            actual = (test[label_column] > thr).astype(int)
            for cut in decision_cuts:
                preds = (probs >= cut).astype(int)
                tp = int(((preds == 1) & (actual == 1)).sum())
                fp = int(((preds == 1) & (actual == 0)).sum())
                tn = int(((preds == 0) & (actual == 0)).sum())
                fn = int(((preds == 0) & (actual == 1)).sum())
                total = tp + fp + tn + fn
                rows.append({
                    "model": name,
                    "target_month": month.strftime("%Y-%m"),
                    "threshold": thr,
                    "decision_cut": cut,
                    "flagged": int(preds.sum()),
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "accuracy": (tp + tn) / total if total else 0.0,
                    "log_loss": logloss,
                    "brier": brier,
                    "roc_auc": roc_auc,
                    "avg_precision": avg_prec,
                    "ece": ece,
                    "gain_mae": gain_mae,
                    "gain_rmse": gain_rmse,
                    "topk_log_loss": topk_log_loss,
                    "topk_brier": topk_brier,
                    "topk_gain_diff": topk_gain_diff,
                    "topk_actual_gain": topk_actual_gain,
                    "topk_expected_gain": topk_expected_gain,
                })
    return pd.DataFrame(rows)


def run_benchmarks(
    *,
    label_column: str = "future_gain",
    growth_threshold: float | None = None,
    thresholds: Tuple[float, ...] | None = None,
    cache_dir: Path | None = None,
    prefix: str = "v3",
    decision_cuts: Tuple[float, ...] | None = None,
    severity_column: str = "future_gain",
    topk: int = 20,
) -> None:
    df = prepare_dataset()
    cache = cache_dir or PATHS.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    split_cfg = build_splits(df, thresholds=thresholds or CONFIG.thresholds)
    models = baseline_models()
    all_results = []
    model_iter = list(models.items())
    if tqdm is not None:
        model_iter = tqdm(model_iter, desc="Benchmark models", unit="model")
    for name, (features, model) in model_iter:
        res = evaluate_model(
            name,
            features,
            model,
            df,
            split_cfg,
            decision_cuts=decision_cuts or CONFIG.decision_cuts,
            label_column=label_column,
            growth_threshold=growth_threshold if growth_threshold is not None else CONFIG.target_growth,
            severity_column=severity_column,
            topk=topk,
        )
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(cache / f"{prefix}_benchmarks_detailed.csv", index=False)

    summary = (
        results.groupby(["model", "threshold", "decision_cut"])
        .agg(mean_precision=("precision", "mean"),
             mean_recall=("recall", "mean"),
             mean_accuracy=("accuracy", "mean"),
             mean_log_loss=("log_loss", "mean"),
             mean_brier=("brier", "mean"),
             mean_roc_auc=("roc_auc", "mean"),
             mean_avg_precision=("avg_precision", "mean"),
             mean_ece=("ece", "mean"),
             mean_gain_mae=("gain_mae", "mean"),
             mean_gain_rmse=("gain_rmse", "mean"),
             mean_topk_log_loss=("topk_log_loss", "mean"),
             mean_topk_brier=("topk_brier", "mean"),
             mean_topk_gain_diff=("topk_gain_diff", "mean"),
             mean_topk_actual_gain=("topk_actual_gain", "mean"),
             mean_topk_expected_gain=("topk_expected_gain", "mean"),
             mean_flagged=("flagged", "mean"),
             mean_tp=("TP", "mean"),
             mean_fp=("FP", "mean"),
             mean_fn=("FN", "mean"),
             mean_tn=("TN", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(cache / f"{prefix}_benchmarks_summary.csv", index=False)
    print(f"Saved {prefix} benchmark results to", cache)


def run_price_benchmarks() -> None:
    run_benchmarks(
        label_column="future_price_gain",
        growth_threshold=CONFIG.price_target_growth,
        thresholds=CONFIG.thresholds,
        cache_dir=PATHS.price_cache_dir,
        prefix="v3_price",
        decision_cuts=CONFIG.price_decision_cuts,
        severity_column="future_price_gain",
    )


if __name__ == "__main__":
    run_benchmarks()
