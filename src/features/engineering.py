"""Shared feature-engineering helpers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, Tuple

import pandas as pd

import numpy as np


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


def add_time_since_spike(
    df: pd.DataFrame,
    *,
    group_col: str = "sa2_code",
    month_col: str = "month",
    rent_col: str = "median_rent",
    threshold: float = 0.02,
    out_col: str = "time_since_spike",
) -> pd.DataFrame:
    """Add months since the last rent spike (> ``threshold`` change) per SA2."""
    out = df.sort_values([group_col, month_col]).copy()
    out["__rent_change__"] = out.groupby(group_col)[rent_col].pct_change(fill_method=None)

    def _calc(group: pd.DataFrame) -> pd.DataFrame:
        change = group["__rent_change__"].to_numpy()
        values = np.full(len(group), np.nan, dtype=float)
        last_idx: int | None = None
        for i, ch in enumerate(change):
            if pd.notna(ch) and ch > threshold:
                last_idx = i
                values[i] = 0.0
            elif last_idx is not None:
                values[i] = float(i - last_idx)
        group[out_col] = values
        group[out_col] = group[out_col].ffill()
        return group

    out = out.groupby(group_col, group_keys=False).apply(_calc)
    fallback = out[out_col].max(skipna=True)
    fallback = float(fallback) if np.isfinite(fallback) else 24.0
    out[out_col] = out[out_col].fillna(fallback).astype(float)
    out = out.drop(columns=["__rent_change__"])
    return out


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
        with np.errstate(invalid="ignore", divide="ignore"):  # guard all-NaN slices
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0, ddof=0)
        # Handle columns that are entirely NaN or constant
        all_nan = np.isnan(X).all(axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where(np.isnan(sd) | (sd == 0.0) | all_nan, 1.0, sd)
    else:
        mu, sd = stats
        sd = np.where(sd == 0.0, 1.0, sd)
        mu = np.where(np.isnan(mu), 0.0, mu)
    X = np.where(np.isnan(X), mu, X)
    Xz = (X - mu) / sd
    return Xz, mu, sd


def apply_standardization(df: pd.DataFrame, columns: Sequence[str], mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply precomputed standardization parameters to ``df`` columns."""
    X = df.loc[:, columns].to_numpy(dtype=float)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where(sd == 0.0, 1.0, sd)
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


def compute_wa_aggregates(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    sa2_col: str = "sa2_code",
    rent_mom_col: str = "rent_mom_1m",
    lodgements_col: str = "count_lodgements",
    disposals_col: str = "count_disposals",
    stock_col: str = "stock_bonds",
) -> pd.DataFrame:
    """Compute WA-level aggregates (rent momentum, churn, stock growth, disposal rate).

    Returns a frame with columns: month, wa_rent_mom, wa_churn_rate,
    wa_stock_growth, wa_disp_rate. Missing inputs yield an empty frame.
    """
    required = {month_col, sa2_col, stock_col}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=[month_col, "wa_rent_mom", "wa_churn_rate", "wa_stock_growth", "wa_disp_rate"])

    # Normalize types and sort the full frame so any optional columns
    # align with the sorted order before assignment (avoids misalignment
    # when the input df arrives unsorted).
    df2 = df.copy()
    df2[month_col] = pd.to_datetime(df2[month_col])
    df2[sa2_col] = df2[sa2_col].astype(str)
    df2 = df2.sort_values([sa2_col, month_col]).reset_index(drop=True)

    work = df2[[month_col, sa2_col, stock_col]].copy()

    # Attach optional inputs if present (already aligned by sorting df2)
    for col in [rent_mom_col, lodgements_col, disposals_col]:
        if col in df2.columns:
            work[col] = df2[col].to_numpy()
        else:
            work[col] = np.nan

    work["stock_prev"] = work.groupby(sa2_col)[stock_col].shift(1)

    # Weighted rent momentum by stock (fallback to 0 when missing)
    # Avoid groupby.apply to keep behavior stable across pandas versions.
    _tmp = work.assign(
        _w=work[stock_col].fillna(0.0),
        _wm=work[stock_col].fillna(0.0) * work[rent_mom_col].fillna(0.0),
    )
    rm = (
        _tmp.groupby(month_col, as_index=False)[["_w", "_wm"]].sum()
        .assign(wa_rent_mom=lambda t: t["_wm"] / t["_w"].replace({0.0: 1.0}))
        [[month_col, "wa_rent_mom"]]
    )

    # Totals for churn / disposals / stock growth
    agg_cols = {
        "count_lodgements": lodgements_col,
        "count_disposals": disposals_col,
        "stock_bonds": stock_col,
        "stock_prev": "stock_prev",
    }
    missing_source = [c for c in agg_cols.values() if c not in work.columns]
    for src in missing_source:
        work[src] = work[src] if src in work.columns else np.nan

    tot = (
        work.groupby(month_col, as_index=False)
        .agg({
            lodgements_col: "sum",
            disposals_col: "sum",
            stock_col: "sum",
            "stock_prev": "sum",
        })
        .rename(columns={lodgements_col: "count_lodgements", disposals_col: "count_disposals"})
    )
    denom = tot[stock_col].replace({0: np.nan})
    tot["wa_churn_rate"] = (tot["count_lodgements"] + tot["count_disposals"]) / denom
    tot["wa_disp_rate"] = tot["count_disposals"] / denom
    tot["wa_stock_growth"] = (tot[stock_col] / tot["stock_prev"].replace({0: np.nan})) - 1.0
    tot[["wa_churn_rate", "wa_disp_rate", "wa_stock_growth"]] = tot[["wa_churn_rate", "wa_disp_rate", "wa_stock_growth"]].replace([np.inf, -np.inf], np.nan)
    tot["wa_churn_rate"] = tot["wa_churn_rate"].fillna(0.0)
    tot["wa_disp_rate"] = tot["wa_disp_rate"].fillna(0.0)
    tot["wa_stock_growth"] = tot["wa_stock_growth"].fillna(0.0)

    out = tot[[month_col, "wa_churn_rate", "wa_disp_rate", "wa_stock_growth"]].merge(rm, on=month_col, how="left")
    return out.sort_values(month_col).reset_index(drop=True)


__all__ = [
    "add_calendar_features",
    "add_rent_momentum",
    "add_churn_proxy",
    "standardize_columns",
    "apply_standardization",
    "ensure_columns",
    "add_interaction_features",
    "add_group_demean",
    "add_time_since_spike",
    "compute_wa_aggregates",
]
