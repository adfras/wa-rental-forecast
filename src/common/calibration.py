"""Calibration helpers for forecast probabilities.

Implements a rolling logit (temperature + intercept) calibrator with automatic
selection against an isotonic fallback using time-series cross-validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

try:  # scikit-learn is optional but preferred
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    IsotonicRegression = None  # type: ignore

EPS = 1e-6


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(float), EPS, 1.0 - EPS)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    q = _safe_prob(p)
    return np.log(q) - np.log1p(-q)


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if weights is None:
        return float(np.mean(values))
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = float(w.sum())
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * w) / total)


def _weighted_brier(y: np.ndarray, p: np.ndarray, weights: Optional[np.ndarray]) -> float:
    dif = (p - y) ** 2
    return _weighted_mean(dif, weights)


def _weighted_log_loss(y: np.ndarray, p: np.ndarray, weights: Optional[np.ndarray]) -> float:
    p = _safe_prob(p)
    loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return _weighted_mean(loss, weights)


@dataclass
class CalibrationModel:
    name: str
    info: dict

    def transform(self, probs: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LogitTemperatureModel(CalibrationModel):
    def __init__(self, bias: float, temperature: float):
        super().__init__(
            name="logit_temperature",
            info={"bias": float(bias), "temperature": float(temperature)},
        )
        self.bias = float(bias)
        self.temperature = float(temperature)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        z = _safe_logit(np.asarray(probs, dtype=float))
        logits = self.bias + self.temperature * z
        return _expit(logits)


class IsotonicShrinkModel(CalibrationModel):
    def __init__(self, iso: IsotonicRegression, shrink: float, base_rate: float):  # type: ignore[valid-type]
        super().__init__(
            name="isotonic_shrink",
            info={"shrink": float(shrink), "base_rate": float(base_rate)},
        )
        self.iso = iso
        self.shrink = float(shrink)
        self.base_rate = float(base_rate)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        x = _safe_prob(np.asarray(probs, dtype=float))
        fitted = self.iso.transform(x)
        if self.shrink < 1.0:
            fitted = self.shrink * fitted + (1.0 - self.shrink) * self.base_rate
        return _safe_prob(fitted)


def _fit_logit_temperature(
    frame: pd.DataFrame,
    prob_col: str,
    target_col: str,
    weight_col: Optional[str] = None,
    *,
    l2: float = 0.01,
    max_iter: int = 75,
    tol: float = 1e-6,
) -> Optional[LogitTemperatureModel]:
    y = frame[target_col].astype(float).to_numpy()
    if np.unique(y).size < 2:
        return None
    x = _safe_logit(frame[prob_col].to_numpy(dtype=float))
    w = None
    if weight_col is not None and weight_col in frame:
        w = frame[weight_col].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    else:
        w = np.ones_like(y)
    bias = 0.0
    temp = 1.0
    for _ in range(max_iter):
        logits = bias + temp * x
        p = _expit(logits)
        diff = p - y
        grad_bias = np.sum(w * diff) + l2 * bias
        grad_temp = np.sum(w * diff * x) + l2 * temp
        w_p = w * p * (1.0 - p)
        h00 = np.sum(w_p) + l2
        h11 = np.sum(w_p * x * x) + l2
        h01 = np.sum(w_p * x)
        det = h00 * h11 - h01 * h01
        if det <= 0.0:
            return None
        inv = np.array([[h11, -h01], [-h01, h00]]) / det
        step = inv @ np.array([grad_bias, grad_temp])
        bias_new = bias - step[0]
        temp_new = temp - step[1]
        if not np.isfinite(bias_new) or not np.isfinite(temp_new):
            return None
        if abs(bias_new - bias) < tol and abs(temp_new - temp) < tol:
            bias, temp = bias_new, temp_new
            break
        bias, temp = bias_new, temp_new
    if not np.isfinite(bias) or not np.isfinite(temp):
        return None
    return LogitTemperatureModel(float(bias), float(temp))


def _fit_isotonic(
    frame: pd.DataFrame,
    prob_col: str,
    target_col: str,
    weight_col: Optional[str] = None,
    *,
    shrink: float = 0.85,
) -> Optional[IsotonicShrinkModel]:
    if IsotonicRegression is None:
        return None
    y = frame[target_col].astype(float).to_numpy()
    if np.unique(y).size < 2:
        return None
    x = _safe_prob(frame[prob_col].to_numpy(dtype=float))
    sample_weight = None
    if weight_col is not None and weight_col in frame:
        sample_weight = frame[weight_col].to_numpy(dtype=float)
        sample_weight = np.where(np.isfinite(sample_weight) & (sample_weight > 0.0), sample_weight, 0.0)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y, sample_weight=sample_weight)
    base_rate = _weighted_mean(y, sample_weight)
    return IsotonicShrinkModel(iso, shrink, base_rate)


@dataclass
class CalibrationSelection:
    model: CalibrationModel
    cv_score: float
    cv_metric: str
    train_months: list[pd.Timestamp]
    candidate_scores: dict


def _iter_cv_splits(months: list[pd.Timestamp], window: int) -> Iterable[tuple[list[pd.Timestamp], pd.Timestamp]]:
    for idx in range(1, len(months)):
        train_months = months[max(0, idx - window): idx]
        if not train_months:
            continue
        yield train_months, months[idx]


def select_calibrator(
    frame: pd.DataFrame,
    *,
    month_col: str = "month",
    prob_col: str = "price_pressure_prob_raw",
    target_col: str = "actual_jump",
    weight_col: Optional[str] = None,
    window: int = 6,
    min_months: int = 4,
    min_rows: int = 200,
    metric: str = "brier",
    prefer: tuple[str, ...] = ("logit_temperature", "isotonic_shrink"),
    l2: float = 0.01,
    shrink: float = 0.85,
) -> Optional[CalibrationSelection]:
    if frame.empty:
        return None
    months = sorted(frame[month_col].dropna().unique())
    if len(months) < min_months:
        return None
    candidates: dict[str, Callable[[pd.DataFrame], Optional[CalibrationModel]]] = {
        "logit_temperature": lambda df: _fit_logit_temperature(df, prob_col, target_col, weight_col, l2=l2),
        "isotonic_shrink": lambda df: _fit_isotonic(df, prob_col, target_col, weight_col, shrink=shrink),
    }
    metric_func: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
    if metric == "logloss":
        metric_func = _weighted_log_loss
    else:
        metric_func = _weighted_brier
    scores: dict[str, list[float]] = {name: [] for name in candidates}
    months_list = [pd.Timestamp(m) for m in months]
    for train_months, val_month in _iter_cv_splits(months_list, window):
        train_idx = frame[month_col].isin(train_months)
        val_idx = frame[month_col] == val_month
        train_df = frame.loc[train_idx]
        val_df = frame.loc[val_idx]
        if len(train_df) < min_rows or val_df.empty:
            continue
        for name, factory in candidates.items():
            model = factory(train_df)
            if model is None:
                continue
            preds = model.transform(val_df[prob_col].to_numpy(dtype=float))
            y = val_df[target_col].to_numpy(dtype=float)
            weights = val_df[weight_col].to_numpy(dtype=float) if weight_col and weight_col in val_df else None
            score = metric_func(y, preds, weights)
            scores[name].append(score)
    # Aggregate scores
    best_name: Optional[str] = None
    best_score = float("inf")
    for name in prefer:
        seq = scores.get(name)
        if not seq:
            continue
        avg = float(np.mean(seq))
        if avg < best_score:
            best_name = name
            best_score = avg
    if best_name is None or not np.isfinite(best_score):
        return None
    # Fit final model on most recent ``window`` months
    tail_months = months_list[-window:]
    train_df = frame[frame[month_col].isin(tail_months)].copy()
    if len(train_df) < min_rows:
        train_df = frame.copy()
    model = candidates[best_name](train_df)
    if model is None:
        return None
    return CalibrationSelection(
        model=model,
        cv_score=best_score,
        cv_metric=metric,
        train_months=[pd.Timestamp(m) for m in sorted(train_df[month_col].unique())],
        candidate_scores={k: float(np.mean(v)) for k, v in scores.items() if v},
    )


def apply_calibration(
    probs: np.ndarray,
    selection: CalibrationSelection | None,
) -> np.ndarray:
    if selection is None:
        return np.asarray(probs, dtype=float)
    return selection.model.transform(np.asarray(probs, dtype=float))


__all__ = [
    "CalibrationModel",
    "CalibrationSelection",
    "IsotonicShrinkModel",
    "LogitTemperatureModel",
    "apply_calibration",
    "select_calibrator",
]
