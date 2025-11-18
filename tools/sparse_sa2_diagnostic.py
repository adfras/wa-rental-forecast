"""Diagnostics for chronically underpredicted sparse SA2s.

Generates summary metrics and per-month series for the thin-data cohort
and saves quick-look plots contrasting actual jumps vs forecasted
probabilities alongside supply proxies.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = BASE_DIR / "outputs" / "tables" / "predictions_with_calibration_full.parquet"
BONDS_PANEL_PATH = BASE_DIR / "data_stage" / "bonds_panel_sa2.parquet"
MERGED_GLOB = BASE_DIR / "outputs" / "tables" / "merged_predictions_full_*.csv"
DEFAULT_SA2_NAMES = ("Roebourne", "Gnowangerup", "Morawa", "City Beach")
PERCENTILE_CUTOFF = 0.95
LOW_LODGEMENT_THRESHOLD = 5
OUTPUT_TABLE_SUMMARY = BASE_DIR / "outputs" / "tables" / "sparse_sa2_summary.csv"
OUTPUT_TABLE_SERIES = BASE_DIR / "outputs" / "tables" / "sparse_sa2_timeseries.csv"
OUTPUT_FIG_DIR = BASE_DIR / "outputs" / "figures"


def _latest_merged_predictions_csv() -> Path:
    matches = sorted(MERGED_GLOB.parent.glob(MERGED_GLOB.name))
    if not matches:
        raise FileNotFoundError("No merged_predictions_full_*.csv files found in outputs/tables")
    return matches[-1]


def _load_sa2_names() -> pd.DataFrame:
    latest = _latest_merged_predictions_csv()
    df = pd.read_csv(latest, usecols=["sa2_code", "sa2_name"])
    return df.drop_duplicates().assign(sa2_code=lambda d: d.sa2_code.astype(str))


def _prepare_timeseries(sa2_names: Sequence[str]) -> pd.DataFrame:
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    bonds = pd.read_parquet(BONDS_PANEL_PATH)
    names = _load_sa2_names()

    merged = (predictions
              .merge(bonds, on=["sa2_code", "month"], how="left")
              .merge(names, on="sa2_code", how="left"))

    mask = merged["sa2_name"].isin(sa2_names)
    timeseries = merged.loc[mask].copy()
    timeseries.sort_values(["sa2_code", "month"], inplace=True)
    timeseries["actual_jump"] = timeseries["actual_jump"].astype(float)
    timeseries["residual"] = timeseries["actual_jump"] - timeseries["price_pressure_prob"]
    timeseries["large_rent_jump"] = (timeseries["rent_momentum"] >= 0.10).astype(int)
    timeseries["pct_change"] = (
        timeseries.groupby("sa2_code")["median_rent"]
        .pct_change(fill_method=None)
        .fillna(0.0)
    )
    timeseries.sort_values(["sa2_name", "month"], inplace=True)
    return timeseries


def _summarise(timeseries: pd.DataFrame) -> pd.DataFrame:
    def pct(series: pd.Series, predicate) -> float:
        return float((predicate(series)).mean()) if not series.empty else float("nan")

    grouped = timeseries.groupby("sa2_name")
    summary = grouped.agg(
        n_months=("month", "count"),
        latest_month=("month", "max"),
        avg_prob=("price_pressure_prob", "mean"),
        avg_calibrated=("prob_calibrated", "mean"),
        jump_rate=("actual_jump", "mean"),
        rmse=("residual", lambda s: (s.pow(2).mean()) ** 0.5),
        avg_residual=("residual", "mean"),
        avg_lodgements=("count_lodgements", "mean"),
        med_lodgements=("count_lodgements", "median"),
        avg_stock=("stock_bonds", "mean"),
        avg_churn=("churn_rate", "mean"),
        avg_net_stock_change=("net_stock_change", "mean"),
        avg_momentum=("rent_momentum", "mean"),
        max_momentum=("rent_momentum", "max"),
        p95_momentum=("rent_momentum", lambda s: s.quantile(PERCENTILE_CUTOFF)),
        avg_pct_change=("pct_change", "mean"),
        max_pct_change=("pct_change", "max"),
    ).reset_index()

    summary["bias"] = summary["jump_rate"] - summary["avg_prob"]

    summary["share_low_lodgements"] = grouped["count_lodgements"].apply(
        lambda s: pct(s, lambda x: x < LOW_LODGEMENT_THRESHOLD)
    ).values
    summary["share_large_momentum"] = grouped["rent_momentum"].apply(
        lambda s: pct(s, lambda x: x >= 0.10)
    ).values
    summary["share_big_pct_change"] = grouped["pct_change"].apply(
        lambda s: pct(s, lambda x: x >= 0.10)
    ).values
    return summary


def _plot_figures(timeseries: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sa2_name, frame in timeseries.groupby("sa2_name"):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(frame["month"], frame["price_pressure_prob"], label="Forecast prob", color="#1f77b4")
        axes[0].plot(frame["month"], frame["prob_calibrated"], label="Calibrated prob", color="#ff7f0e")
        axes[0].step(frame["month"], frame["actual_jump"], where="mid", label="Actual jump", color="#2ca02c")
        axes[0].set_ylabel("Jump prob / Actual")
        axes[0].legend(loc="upper left")

        axes[1].plot(frame["month"], frame["count_lodgements"], marker="o", color="#9467bd")
        axes[1].axhline(LOW_LODGEMENT_THRESHOLD, color="#d62728", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Lodgements")

        axes[2].plot(frame["month"], frame["rent_momentum"], marker="o", color="#8c564b", label="Rent momentum")
        axes[2].plot(frame["month"], frame["pct_change"], marker="x", color="#17becf", label="Median rent Δ")
        axes[2].axhline(0.02, color="#d62728", linestyle=":", linewidth=1)
        axes[2].set_ylabel("Rent change")
        axes[2].legend(loc="upper left")

        fig.suptitle(f"Sparse SA2 diagnostic: {sa2_name}")
        axes[-1].set_xlabel("Month")
        fig.autofmt_xdate()

        out_path = output_dir / f"sparse_sa2_{sa2_name.lower().replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths.append(out_path)
    return paths


def run(sa2_names: Sequence[str]) -> None:
    timeseries = _prepare_timeseries(sa2_names)
    if timeseries.empty:
        raise ValueError("No rows found for the requested SA2 names")

    summary = _summarise(timeseries)
    OUTPUT_TABLE_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_TABLE_SUMMARY, index=False)

    timeseries.to_csv(OUTPUT_TABLE_SERIES, index=False)

    figure_paths = _plot_figures(timeseries, OUTPUT_FIG_DIR)

    print("Summary written to:", OUTPUT_TABLE_SUMMARY)
    print("Timeseries written to:", OUTPUT_TABLE_SERIES)
    print("Figures:")
    for path in figure_paths:
        print(" -", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose sparse SA2 residuals")
    parser.add_argument(
        "--sa2",
        dest="sa2_names",
        nargs="*",
        default=DEFAULT_SA2_NAMES,
        help="SA2 names to include (defaults to known sparse cohort)",
    )
    args = parser.parse_args()
    run(tuple(args.sa2_names))


if __name__ == "__main__":
    main()
