"""Date-utility helpers shared across the modelling pipeline."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Mapping

import numpy as np
import pandas as pd


def to_month(values, *, errors: str = "raise") -> pd.Series | pd.Timestamp:
    """Normalize timestamps to month-level timestamps.

    Parameters
    ----------
    values: scalar, Series, or array-like convertible via ``pd.to_datetime``.
    errors: passed through to :func:`pandas.to_datetime` to control coercion.
    """
    dt = pd.to_datetime(values, errors=errors)
    if hasattr(dt, "dt"):
        return dt.dt.to_period("M").dt.to_timestamp()
    return dt.to_period("M").to_timestamp()


def parse_ym(s: str) -> pd.Timestamp:
    """Parse a YYYY-MM string into the first day of that month."""
    try:
        return pd.to_datetime(s).to_period("M").to_timestamp()
    except Exception as exc:  # pragma: no cover - mirrors argparse behavior
        raise ValueError(f"Invalid YYYY-MM value: {s}") from exc


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return inclusive month sequence between two timestamps (order preserved)."""
    if end < start:
        return []
    periods = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    return [p.to_timestamp() for p in periods]


def prev_month(month: pd.Timestamp) -> pd.Timestamp:
    """Return the month immediately preceding ``month``."""
    return (month.to_period("M") - 1).to_timestamp()


def build_month_index(months: Iterable[pd.Timestamp]) -> dict[pd.Timestamp, int]:
    """Map each month to its ordinal index (sorted ascending)."""
    normalized = {to_month(m) for m in months}
    ordered = sorted(normalized)
    return {m: i for i, m in enumerate(ordered)}


def ensure_month_index(
    months: Iterable[pd.Timestamp],
    existing: Mapping[pd.Timestamp, int] | None = None,
) -> tuple[dict[pd.Timestamp, int], np.ndarray]:
    """Return an index mapping and corresponding integer array for ``months``.

    If ``existing`` is provided, it is reused and must contain every month
    encountered; otherwise a new mapping is constructed.
    """
    if existing is None:
        mapping = build_month_index(months)
    else:
        mapping = dict(existing)
    normalized = to_month(pd.Series(list(months)))
    idx = np.array([mapping[m] for m in normalized], dtype=int)
    return mapping, idx


def compute_recency_weights(months, half_life: float | None) -> np.ndarray | None:
    """Half-life downweighting for chronological observations.

    ``months`` can be a Series or array-like; weights decay by half every
    ``half_life`` steps (interpreted as months). ``None`` is returned when
    ``half_life`` is not positive.
    """
    if half_life is None or half_life <= 0:
        return None
    month_periods = to_month(months, errors="coerce")
    valid = month_periods.notna() if hasattr(month_periods, "notna") else True
    if hasattr(month_periods, "__iter__") and not isinstance(month_periods, pd.Timestamp):
        mp = month_periods[valid]
    else:
        mp = month_periods
    # Normalize to pandas Timestamp for safe dictionary lookups
    mp_index = pd.to_datetime(pd.Index(mp))
    unique = mp_index.unique().sort_values()
    lookup = {pd.Timestamp(m): i for i, m in enumerate(unique)}
    indices = np.array([lookup[pd.Timestamp(m)] for m in mp_index], dtype=float)
    i_max = float(np.max(indices)) if len(indices) else 0.0
    weights = np.empty_like(indices, dtype=float)
    weights[:] = 0.5 ** ((i_max - indices) / float(half_life))

    # Re-insert NaN positions (if any) as 1.0 to avoid zero-weighting rows that lack a month
    if hasattr(month_periods, "notna"):
        full = np.ones(len(month_periods), dtype=float)
        full[valid.values if hasattr(valid, "values") else valid] = weights
        return full
    return weights


__all__ = [
    "to_month",
    "parse_ym",
    "month_range",
    "prev_month",
    "build_month_index",
    "ensure_month_index",
    "compute_recency_weights",
]
