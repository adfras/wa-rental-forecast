"""Shared feature-engineering helpers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    sin_col: str = "mo_sin",
    cos_col: str = "mo_cos",
) -> pd.DataFrame:
    """Add lightweight calendar seasonality and flags to ``df``.

    The month column must be coercible via ``pd.to_datetime``. New columns
    include `mo_sin`, `mo_cos`, `quarter`, `is_eofy`, `is_uni_sem_feb`, and
    `is_uni_sem_jul`.
    """
    dt = pd.to_datetime(df[month_col])
    m = dt.dt.month.astype(int)
    out = df.copy()
    out[sin_col] = np.sin(2 * np.pi * m / 12.0)
    out[cos_col] = np.cos(2 * np.pi * m / 12.0)
    out["quarter"] = dt.dt.quarter.astype(int)
    out["is_eofy"] = (m == 6).astype(int)
    out["is_uni_sem_feb"] = (m == 2).astype(int)
    out["is_uni_sem_jul"] = (m == 7).astype(int)
    return out


def add_rent_momentum(
    df: pd.DataFrame,
    *,
    group_col: str = "sa2_code",
    month_col: str = "month",
    rent_col: str = "median_rent",
) -> pd.DataFrame:
    """Attach 1-month and 3-month rent momentum columns."""
    df_sorted = df.sort_values([group_col, month_col]).copy()
    df_sorted["rent_mom_1m"] = df_sorted.groupby(group_col)[rent_col].pct_change(1, fill_method=None)
    df_sorted["rent_mom_3m"] = (
        df_sorted[rent_col] / df_sorted.groupby(group_col)[rent_col].shift(3)
    ) ** (1 / 3) - 1
    return df_sorted


def add_churn_proxy(
    df: pd.DataFrame,
    *,
    disposals_col: str = "count_disposals",
    lodgements_col: str = "count_lodgements",
    stock_col: str = "stock_bonds",
    days_held_col: str = "mean_days_held",
    out_col: str = "churn_rate",
) -> pd.DataFrame:
    """Compute a churn-rate proxy from lodgements/disposals or fallback columns."""
    out = df.copy()
    cols = set(out.columns)
    if {disposals_col, stock_col} <= cols:
        denom = out[stock_col].replace({0: np.nan})
        out[out_col] = out[disposals_col] / denom
    elif days_held_col in cols:
        out[out_col] = 30.0 / out[days_held_col]
    else:
        out[out_col] = np.nan
    return out


def standardize_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    stats: Tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized numpy array along with (mean, std) statistics.

    If ``stats`` is provided, it must be a tuple of (mean, std) arrays to
    reuse when transforming new data.
    """
    X = df.loc[:, columns].to_numpy(dtype=float)
    if stats is None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
        sd[sd == 0] = 1.0
    else:
        mu, sd = stats
    X = np.where(np.isnan(X), mu, X)
    Xz = (X - mu) / sd
    return Xz, mu, sd


def apply_standardization(df: pd.DataFrame, columns: Sequence[str], mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply precomputed standardization parameters to ``df`` columns."""
    X = df.loc[:, columns].to_numpy(dtype=float)
    X = np.where(np.isnan(X), mu, X)
    return (X - mu) / sd


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise ``SystemExit`` if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")


def add_interaction_features(
    df: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    suffix: str = "__x__",
) -> pd.DataFrame:
    """Add pairwise interaction columns for the provided feature pairs."""
    out = df.copy()
    for a, b in pairs:
        if a not in out.columns or b not in out.columns:
            continue
        name = f"{a}{suffix}{b}"
        out[name] = out[a] * out[b]
    return out


def add_group_demean(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    columns: Sequence[str],
    *,
    suffix: str = "_demeaned",
) -> pd.DataFrame:
    """Add demeaned variants of columns based on group means."""
    out = df.copy()
    means = out.groupby(list(group_cols))[list(columns)].transform("mean")
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}{suffix}"] = out[col] - means[col]
    return out


__all__ = [
    "add_calendar_features",
    "add_rent_momentum",
    "add_churn_proxy",
    "standardize_columns",
    "apply_standardization",
    "ensure_columns",
    "add_interaction_features",
    "add_group_demean",
]
