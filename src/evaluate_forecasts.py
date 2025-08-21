# src/evaluate_forecasts.py
# Evaluate forecast accuracy and export named spreadsheets (clean + robust).
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from src.config import (
    STAGE_DIR,
    OUT_DIR,
    FIG_DIR,
    ASGS_SA2_GPKG,
    RENT_GROWTH_THRESHOLD,
)

# -------------------- small utilities --------------------

def _to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()

def _first_present(cols: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not find any of {candidates}; columns are {cols[:12]} ...")

def _find_sa2_layer(path: Path) -> str:
    """Pick the SA2 layer from the ASGS GeoPackage (pyogrio if available)."""
    try:
        from pyogrio import list_layers
        names = [n[0] if isinstance(n, (list, tuple)) else n for n in list_layers(path)]
        # prefer canonical SA2 2021 names
        for prefer in ("SA2_2021_AUST_GDA2020", "SA2_2021_AUST_GDA94"):
            if prefer in names:
                return prefer
        # otherwise first that mentions SA2
        for n in names:
            if "SA2" in str(n).upper():
                return n
        return names[0]
    except Exception:
        # sensible default most files use
        return "SA2_2021_AUST_GDA2020"

def _read_sa2_names() -> pd.DataFrame:
    """Return a 2-column frame: sa2_code, sa2_name (name may be missing)."""
    layer = _find_sa2_layer(ASGS_SA2_GPKG)
    gdf = gpd.read_file(ASGS_SA2_GPKG, layer=layer)

    cols = list(gdf.columns)
    code_col = _first_present(cols, ["SA2_MAINCODE_2021", "SA2_CODE_2021", "SA2_CODE21"])
    name_col = None
    for c in ["SA2_NAME_2021", "SA2_NAME21"]:
        if c in cols:
            name_col = c
            break

    keep = [code_col] + ([name_col] if name_col else [])
    out = gdf[keep].copy()
    out.rename(columns={code_col: "sa2_code"}, inplace=True)
    out["sa2_code"] = out["sa2_code"].astype(str)
    if name_col:
        out.rename(columns={name_col: "sa2_name"}, inplace=True)
        out["sa2_name"] = out["sa2_name"].astype(str)
    return out

def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via rank-sum; handles ties."""
    n = len(y)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    uniq, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        means = np.bincount(inv, ranks) / cnt[inv]
        ranks = means
    n1 = float(y.sum()); n0 = float(n - n1)
    if n0 == 0 or n1 == 0:
        return np.nan
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def _calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    # qcut drops duplicate edges automatically where needed
    d["bin"] = pd.qcut(d["p"], q=min(bins, d["p"].nunique()), duplicates="drop")
    g = d.groupby("bin", observed=True)
    out = g.agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out

# -------------------- labels from realized rents --------------------

def _build_realized_labels(threshold: float) -> pd.DataFrame:
    """Return sa2_code, month, actual_jump for all months with a previous month."""
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    sa2["month"] = _to_month(sa2["month"])
    sa2["sa2_code"] = sa2["sa2_code"].astype(str)
    sa2 = sa2.sort_values(["sa2_code", "month"])
    sa2["rent_prev"] = sa2.groupby("sa2_code")["median_rent"].shift(1)
    sa2["actual_jump"] = (sa2["median_rent"] / sa2["rent_prev"] - 1.0) > threshold
    return sa2.loc[sa2["rent_prev"].notna(), ["sa2_code", "month", "actual_jump"]]

# -------------------- main entrypoint --------------------

def main(threshold: float = RENT_GROWTH_THRESHOLD):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Latest predictions (write plain + named spreadsheets)
    preds = pd.read_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet").copy()
    preds["month"] = _to_month(preds["month"])
    preds["sa2_code"] = preds["sa2_code"].astype(str)

    latest_month = preds["month"].max()
    latest = preds.loc[preds["month"] == latest_month].copy()
    latest.sort_values("price_pressure_prob", ascending=False).to_excel(
        OUT_DIR / "price_pressure_forecast_sa2_latest.xlsx", index=False
    )

    # Named version (join SA2 names from ASGS)
    names = _read_sa2_names()   # sa2_code, optional sa2_name
    named = latest.merge(names, on="sa2_code", how="left")
    cols = ["sa2_code"] + (["sa2_name"] if "sa2_name" in named.columns else []) + ["month", "price_pressure_prob"]
    named[cols].to_excel(OUT_DIR / "price_pressure_forecast_sa2_latest_named.xlsx", index=False)
    print(f"Wrote {OUT_DIR / 'price_pressure_forecast_sa2_latest.xlsx'}")
    print(f"Wrote {OUT_DIR / 'price_pressure_forecast_sa2_latest_named.xlsx'} ({len(named)} rows for {latest_month:%Y-%m}).")

    # 2) Score predictions where realized outcomes exist
    labels = _build_realized_labels(threshold)
    joined = preds.merge(labels, on=["sa2_code", "month"], how="inner")
    if joined.empty:
        print("No realized months yet to score. (Run again once the next bond ZIP is ingested.)")
        return

    # Summary metrics per month
    summary_rows = []
    for m, dfm in joined.groupby("month"):
        y = dfm["actual_jump"].astype(int).to_numpy()
        p = dfm["price_pressure_prob"].astype(float).to_numpy()

        thr = 0.5
        yhat = (p >= thr).astype(int)

        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())

        summary_rows.append({
            "month": m,
            "n_sa2": len(dfm),
            "base_rate": float(y.mean()),
            "auc": _roc_auc(y, p),
            "brier": _brier(y, p),
            "log_loss": _log_loss(y, p),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision_at_0.5": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall_at_0.5": tp / (tp + fn) if (tp + fn) else np.nan,
            "accuracy_at_0.5": (tp + tn) / len(dfm) if len(dfm) else np.nan,
        })

    summary = pd.DataFrame(summary_rows).sort_values("month")
    summary_path = OUT_DIR / "forecast_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # 3) Detailed & calibration for latest scored month
    latest_scored = summary["month"].max()
    detail = joined.loc[joined["month"] == latest_scored].copy()
    det_path = OUT_DIR / f"forecast_eval_details_{latest_scored:%Y-%m}.csv"
    detail.to_csv(det_path, index=False)
    print(f"Wrote {det_path}")

    cal = _calibration_bins(
        detail["actual_jump"].astype(int).to_numpy(),
        detail["price_pressure_prob"].astype(float).to_numpy(),
        bins=10,
    )
    cal_path = OUT_DIR / f"forecast_calibration_{latest_scored:%Y-%m}.csv"
    cal.to_csv(cal_path, index=False)
    print(f"Wrote {cal_path}")

    # Calibration plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(cal["pred"], cal["actual"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration — {latest_scored:%Y-%m}")
    fig.tight_layout()
    png = FIG_DIR / f"forecast_calibration_{latest_scored:%Y-%m}.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    print(f"Wrote {png}")

if __name__ == "__main__":
    main()
