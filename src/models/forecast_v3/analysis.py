"""Exploratory summaries for forecast v3."""
from __future__ import annotations

import pandas as pd

from .config import CONFIG, PATHS
from .data import prepare_dataset


def monthly_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, g in df.groupby("target_month"):
        if g.empty:
            continue
        for thr in CONFIG.thresholds:
            future = g["future_gain"].dropna()
            if future.empty:
                continue
            rises = (future > thr).sum()
            rows.append({
                "target_month": month.strftime("%Y-%m"),
                "threshold": thr,
                "n_obs": len(g),
                "actual_rises": int(rises),
                "base_rate": float(rises / len(g)),
            })
    return pd.DataFrame(rows).sort_values(["target_month", "threshold"])


def feature_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    stats = df[columns].describe(percentiles=[0.1, 0.5, 0.9]).transpose()
    stats = stats.rename(columns={
        "50%": "median",
        "10%": "p10",
        "90%": "p90",
    })
    stats.reset_index(inplace=True)
    stats = stats.rename(columns={"index": "feature"})
    return stats


def run_analysis() -> None:
    df = prepare_dataset()
    target_summary = monthly_target_summary(df)
    target_summary.to_csv(PATHS.cache_dir / "v3_target_summary.csv", index=False)

    feature_cols = [
        "median_rent",
        "availability_rate",
        "rent_change_1m",
        "rent_change_3m",
        "churn_rate",
        "net_stock_change",
        "stock_bonds",
        "mean_days_held",
        "hp_median_price",
        "wa_unemp_rate_sa",
        "wa_bonds_rent_index",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    feat_summary = feature_summary(df, feature_cols)
    feat_summary.to_csv(PATHS.cache_dir / "v3_feature_summary.csv", index=False)
    print("Saved analysis outputs to", PATHS.cache_dir)


if __name__ == "__main__":
    run_analysis()
