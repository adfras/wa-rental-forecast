"""Helpers for building negative-binomial nowcast inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.features.dates import build_month_index, to_month


@dataclass
class NowcastDesign:
    """Arrays and metadata required to fit the nowcast model."""

    frame: pd.DataFrame
    y: np.ndarray
    log_stock: np.ndarray
    time_index: np.ndarray
    season_index: np.ndarray
    sa2_codes: pd.Index
    sa2_index: np.ndarray
    month_index: Mapping[pd.Timestamp, int]

    @property
    def n_sa2(self) -> int:
        return len(self.sa2_codes)

    @property
    def n_obs(self) -> int:
        return len(self.y)


def prepare_design(
    df: pd.DataFrame,
    *,
    month_index: Mapping[pd.Timestamp, int] | None = None,
    response_col: str = "count_disposals",
    exposure_col: str = "stock_bonds",
    group_col: str = "sa2_code",
    month_col: str = "month",
    dropna_exposure: bool = True,
) -> NowcastDesign:
    """Return standardized design arrays for the NB nowcast.

    Parameters
    ----------
    df : DataFrame
        Source data containing response, exposure, group, and month columns.
    month_index : optional mapping of month Timestamp → integer. If omitted,
        it is built from ``df`` (sorted ascending).
    dropna_exposure : whether to drop rows with missing exposure values.
    """
    work = df.copy()
    work[group_col] = work[group_col].astype(str)
    work[month_col] = to_month(work[month_col])

    if dropna_exposure:
        work = work.dropna(subset=[exposure_col])

    work = work.sort_values([group_col, month_col]).reset_index(drop=True)

    if month_index is None:
        month_index = build_month_index(work[month_col].unique())
    else:
        # ensure provided mapping covers all months
        missing = [m for m in work[month_col].unique() if m not in month_index]
        if missing:
            raise SystemExit(f"Month index missing entries for: {missing[:5]}")

    y = work[response_col].fillna(0).astype(int).to_numpy()
    exposure = work[exposure_col].clip(lower=1.0).to_numpy()
    log_stock = np.log(exposure)
    time_index = work[month_col].map(month_index).astype(int).to_numpy()
    season_index = work[month_col].dt.month.astype(int).to_numpy() - 1

    sa2_codes = pd.Index(sorted(work[group_col].unique()))
    sa2_cat = pd.Categorical(work[group_col], categories=sa2_codes)
    sa2_index = sa2_cat.codes.astype(int)

    return NowcastDesign(
        frame=work,
        y=y,
        log_stock=log_stock,
        time_index=time_index,
        season_index=season_index,
        sa2_codes=sa2_codes,
        sa2_index=sa2_index,
        month_index=dict(month_index),
    )


__all__ = ["NowcastDesign", "prepare_design"]
