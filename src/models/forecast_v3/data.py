"""Data loading helpers for forecast v3."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import PATHS, CONFIG


def _load_bonds_panel() -> pd.DataFrame:
    df = pd.read_parquet(PATHS.stage_dir / "bonds_panel_sa2.parquet")
    df["month"] = pd.to_datetime(df["month"])
    return df


def _load_availability() -> pd.DataFrame:
    df = pd.read_parquet(PATHS.stage_dir / "availability_nowcast_sa2.parquet")
    df["month"] = pd.to_datetime(df["month"])
    return df


def _load_external_signals() -> pd.DataFrame:
    path = PATHS.stage_dir / "external_signals.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["month"])
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


def _load_macro_signals() -> pd.DataFrame:
    path = PATHS.macro_signals
    if not path.exists():
        return pd.DataFrame(columns=["month"])
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


def _load_airbnb_metrics() -> pd.DataFrame:
    path = PATHS.airbnb_metrics
    if not path.exists():
        return pd.DataFrame(columns=["sa2_code", "month"])
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df


def _load_text_embeddings() -> pd.DataFrame:
    path = PATHS.text_embeddings
    if not path.exists():
        return pd.DataFrame(columns=["sa2_code", "month"])
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df


def _load_image_embeddings() -> pd.DataFrame:
    path = PATHS.image_embeddings
    if not path.exists():
        return pd.DataFrame(columns=["sa2_code", "month"])
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df


def _load_house_prices() -> pd.DataFrame:
    path = PATHS.stage_dir / "house_prices_sa2_monthly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["sa2_code", "month"])
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df


def _load_house_price_labels() -> pd.DataFrame:
    path = PATHS.house_price_labels
    if not path.exists():
        return pd.DataFrame(columns=["sa2_code", "month", "future_price_gain"])
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df


def _load_spatial_neighbors() -> pd.DataFrame:
    path = PATHS.spatial_neighbors
    if not path.exists():
        return pd.DataFrame(columns=["sa2_code", "neighbor_sa2", "weight"])
    df = pd.read_parquet(path)
    df["sa2_code"] = df["sa2_code"].astype(str)
    df["neighbor_sa2"] = df["neighbor_sa2"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_base_panel() -> pd.DataFrame:
    bonds = _load_bonds_panel()
    avail = _load_availability()
    panel = bonds.merge(avail, on=["sa2_code", "month"], how="left")

    ext = _load_external_signals()
    if not ext.empty:
        panel = panel.merge(ext, on="month", how="left")

    macro = _load_macro_signals()
    if not macro.empty:
        panel = panel.merge(macro, on="month", how="left")

    airbnb = _load_airbnb_metrics()
    if not airbnb.empty:
        panel = panel.merge(airbnb, on=["sa2_code", "month"], how="left")

    hp = _load_house_prices()
    if not hp.empty:
        hp = hp.rename(columns={"median_house_price": "hp_median_price"})
        panel = panel.merge(hp[["sa2_code", "month", "hp_median_price"]], on=["sa2_code", "month"], how="left")

    hp_labels = _load_house_price_labels()
    if hp_labels.empty and not hp.empty:
        hp_labels = hp[["sa2_code", "month", "hp_median_price"]].copy()
        hp_labels.sort_values(["sa2_code", "month"], inplace=True)
        grouped = hp_labels.groupby("sa2_code", sort=False)
        next_price = grouped["hp_median_price"].shift(-1)
        hp_labels["future_price_gain"] = (next_price / hp_labels["hp_median_price"]) - 1.0
    if not hp_labels.empty:
        keep_cols = [c for c in hp_labels.columns if c not in {"hp_median_price"}]
        panel = panel.merge(hp_labels[keep_cols], on=["sa2_code", "month"], how="left")

    text_emb = _load_text_embeddings()
    if not text_emb.empty:
        panel = panel.merge(text_emb, on=["sa2_code", "month"], how="left")

    img_emb = _load_image_embeddings()
    if not img_emb.empty:
        panel = panel.merge(img_emb, on=["sa2_code", "month"], how="left", suffixes=("", "_img"))

    panel = panel.sort_values(["sa2_code", "month"]).reset_index(drop=True)
    return panel


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sa2_code"] = out["sa2_code"].astype(str)
    out["month_idx"] = out["month"].dt.month
    out["year"] = out["month"].dt.year

    # Cyclical month encoding
    out["month_sin"] = np.sin(2 * np.pi * out["month_idx"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_idx"] / 12)

    # Rent change features
    group = out.groupby("sa2_code", sort=False)
    out["rent_change_1m"] = (
        group["median_rent"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    out["rent_change_3m"] = (
        group["median_rent"].pct_change(3, fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    out["availability_diff"] = group["availability_rate"].diff()
    out["stock_change_pct"] = group["stock_bonds"].pct_change(fill_method=None)

    # Airbnb derived metrics (if present)
    if {"listing_count", "avg_daily_rate", "occupancy_rate"}.issubset(out.columns):
        out["airbnb_density"] = out["listing_count"] / out.groupby("month")["listing_count"].transform("sum")
        out["airbnb_revpar"] = out["avg_daily_rate"] * out["occupancy_rate"]

    # Interaction examples
    out["avail_x_churn"] = out["availability_rate"] * out["churn_rate"]
    out["rent_change_1m_x_avail"] = out["rent_change_1m"] * out["availability_rate"]

    # Macro lagged changes if available
    macro_cols = [c for c in out.columns if c.startswith("macro_") or c in {"wa_unemp_rate_sa", "perth_cpi", "m2_growth"}]
    group_month = out.groupby("sa2_code", sort=False)
    for col in macro_cols:
        out[f"{col}_diff_1m"] = group_month[col].diff()
        out[f"{col}_yoy"] = group_month[col].pct_change(12, fill_method=None)

    # Lagged features based on config
    for lag in CONFIG.feature_lags:
        out[f"rent_change_lag_{lag}"] = group["median_rent"].pct_change(lag, fill_method=None)
        out[f"availability_rate_lag_{lag}"] = group["availability_rate"].shift(lag)
        out[f"churn_rate_lag_{lag}"] = group["churn_rate"].shift(lag)
        if "airbnb_density" in out.columns:
            out[f"airbnb_density_lag_{lag}"] = group["airbnb_density"].shift(lag)

    # Spatial lag features
    neighbors = _load_spatial_neighbors()
    if not neighbors.empty:
        weights = neighbors.copy()
        weights["weight"] = weights.get("weight", 1.0)
        merged = out.merge(weights, on="sa2_code", how="left")
        for col in ["median_rent", "availability_rate", "airbnb_density"]:
            if col in out.columns:
                merged[f"{col}_neighbor_weighted"] = merged[col] * merged["weight"]
                neighbor_sum = merged.groupby(["sa2_code", "month"], sort=False)[f"{col}_neighbor_weighted"].sum()
                weight_sum = merged.groupby(["sa2_code", "month"], sort=False)["weight"].sum().replace(0, np.nan)
                out[f"{col}_spatial_mean"] = neighbor_sum / weight_sum

    # Target month (next month)
    out["target_month"] = out["month"] + pd.offsets.MonthBegin(1)
    out["future_gain"] = (
        group["median_rent"].shift(-1) / out["median_rent"] - 1.0
    )

    # House price forward gain / classification (if available)
    if "hp_median_price" in out.columns:
        group_price = out.groupby("sa2_code", sort=False)
        out["future_price_gain"] = (
            group_price["hp_median_price"].shift(-1) / out["hp_median_price"] - 1.0
        )
        out["future_price_gain"] = out["future_price_gain"].replace([np.inf, -np.inf], np.nan)
    if "future_price_gain" in out.columns:
        out["price_jump_target"] = (
            out["future_price_gain"] >= CONFIG.price_target_growth
        ).astype("Int8")
    return out


def prepare_dataset() -> pd.DataFrame:
    df = load_base_panel()
    df = add_basic_features(df)
    if CONFIG.min_history_month:
        df = df[df["month"] >= pd.Timestamp(CONFIG.min_history_month)]
    if CONFIG.max_history_month:
        df = df[df["month"] <= pd.Timestamp(CONFIG.max_history_month)]
    df = df.sort_values(["sa2_code", "month"]).reset_index(drop=True)
    return df
