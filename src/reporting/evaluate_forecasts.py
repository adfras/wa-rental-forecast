"""
Evaluate forecast accuracy and export named spreadsheets and figures.

Outputs include monthly summary metrics (AUC, Brier, log-loss), per-month
details, calibration CSVs and plots, and named spreadsheets that join SA2
names from the ASGS GeoPackage.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.sa2 import load_sa2_names
from src.config import (
    STAGE_DIR,
    OUT_DIR,
    FIG_DIR,
    EVAL_DIR,
    ASGS_SA2_GPKG,
    RENT_GROWTH_THRESHOLD,
)
from src.features.dates import to_month

# -------------------- small utilities --------------------

def _read_sa2_names() -> pd.DataFrame:
    """Return a 2-column frame: sa2_code, sa2_name (name may be missing)."""
    return load_sa2_names(ASGS_SA2_GPKG)

def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via rank-sum; handles ties."""
    n = len(y)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    uniq, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        # Compute mean rank per unique probability value, then map back
        sum_ranks = np.bincount(inv, weights=ranks, minlength=len(uniq))
        mean_ranks = sum_ranks / cnt
        ranks = mean_ranks[inv]
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

def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    d = _calibration_bins(y, p, bins)
    if d.empty:
        return float("nan")
    w = d["n"].to_numpy().astype(float)
    conf = d["pred"].to_numpy().astype(float)
    acc = d["actual"].to_numpy().astype(float)
    wsum = float(w.sum()) if float(w.sum()) > 0 else 1.0
    return float(np.sum(w * np.abs(acc - conf)) / wsum)

def _brier_decompose(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, float, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    d = _calibration_bins(y, p, bins)
    if d.empty:
        return float("nan"), float("nan"), float("nan")
    pi = float(y.mean())
    w = d["n"].to_numpy().astype(float)
    wsum = float(w.sum()) if float(w.sum()) > 0 else 1.0
    acc = d["actual"].to_numpy().astype(float)
    conf = d["pred"].to_numpy().astype(float)
    reliability = float(np.sum(w * (conf - acc) ** 2) / wsum)
    resolution = float(np.sum(w * (acc - pi) ** 2) / wsum)
    uncertainty = float(pi * (1 - pi))
    return reliability, resolution, uncertainty


def _threshold_suffix(threshold: float) -> str:
    formatted = f"{threshold:.3f}".rstrip("0").rstrip(".")
    if not formatted:
        formatted = "0"
    return f"thr{formatted.replace('.', 'p')}"

def _prc_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score  # type: ignore
        ap = float(average_precision_score(y, p))
        prec, rec, _ = precision_recall_curve(y, p)
        return ap, rec, prec
    except Exception:
        return float("nan"), np.array([]), np.array([])

def _latest_hp_on_or_before(hp: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Return per-SA2 latest house-price record with hp.month <= month.

    If none exist for any SA2, returns the overall latest per-SA2 record.
    Expects columns: sa2_code, month, median_house_price (plus optional
    price_allocation_weight_sum, price_suburb_count).
    """
    if hp.empty:
        return hp
    hp = hp.copy()
    hp["month"] = to_month(hp["month"])  # ensure Timestamp
    sub = hp[hp["month"] <= month].copy()
    if sub.empty:
        sub = hp.copy()
    sub = sub.sort_values(["sa2_code", "month"]).groupby("sa2_code", as_index=False).tail(1)
    return sub[["sa2_code", "median_house_price", "month"]]

# -------------------- labels from realized rents --------------------

def _build_realized_labels(threshold: float) -> pd.DataFrame:
    """Return sa2_code, month, actual_jump for all months with a previous month."""
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    sa2["month"] = to_month(sa2["month"])
    sa2["sa2_code"] = sa2["sa2_code"].astype(str)
    sa2 = sa2.sort_values(["sa2_code", "month"])
    sa2["rent_prev"] = sa2.groupby("sa2_code")["median_rent"].shift(1)
    sa2["actual_jump"] = (sa2["median_rent"] / sa2["rent_prev"] - 1.0) > threshold
    return sa2.loc[sa2["rent_prev"].notna(), ["sa2_code", "month", "actual_jump"]]

# -------------------- main entrypoint --------------------



def _build_realized_labels_v2(threshold: float) -> pd.DataFrame:
    """Return realized labels and helpful baselines/weights per SA2×month.

    Columns returned:
      - sa2_code, month
      - actual_jump: bool for Δrent > threshold
      - actual_jump_prev: previous month's actual_jump (LOCF baseline)
      - actual_jump_prev_year: same month last year actual_jump (seasonal baseline)
      - stock_bonds: stock at current month (for WA-weighted errors)
    """
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    sa2["month"] = to_month(sa2["month"])
    sa2["sa2_code"] = sa2["sa2_code"].astype(str)
    sa2 = sa2.sort_values(["sa2_code", "month"]).reset_index(drop=True)

    # Actual jump this month (needs previous rent)
    sa2["rent_prev"] = sa2.groupby("sa2_code")["median_rent"].shift(1)
    sa2["actual_jump"] = (sa2["median_rent"] / sa2["rent_prev"] - 1.0) > threshold

    # Baseline labels for previous month and previous year
    sa2["actual_jump_prev"] = sa2.groupby("sa2_code")["actual_jump"].shift(1)
    sa2["actual_jump_prev_year"] = sa2.groupby("sa2_code")["actual_jump"].shift(12)

    # Keep rows with a defined actual (i.e., where we had a previous rent)
    keep = sa2["rent_prev"].notna()
    return sa2.loc[keep, [
        "sa2_code", "month", "actual_jump", "actual_jump_prev", "actual_jump_prev_year", "stock_bonds"
    ]]
def main(threshold: float = RENT_GROWTH_THRESHOLD,
         start: str | None = None,
         end: str | None = None):
    """Run evaluation and write outputs.

    - threshold: rent growth cutoff for binary label (default from config)
    - start/end: optional inclusive month window filters (YYYY-MM)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Predictions — prefer full history if available (suffix-aware)
    suffix = None if math.isclose(threshold, RENT_GROWTH_THRESHOLD) else _threshold_suffix(threshold)

    hist_candidates: list[Path] = []
    if suffix:
        hist_candidates.append(STAGE_DIR / f"price_pressure_forecast_sa2_history_{suffix}.parquet")
    hist_candidates.append(STAGE_DIR / "price_pressure_forecast_sa2_history.parquet")

    preds_path: Path | None = next((p for p in hist_candidates if p.exists()), None)
    latest_candidates: list[Path] = []
    if suffix:
        latest_candidates.append(STAGE_DIR / f"price_pressure_forecast_sa2_{suffix}.parquet")
    latest_candidates.append(STAGE_DIR / "price_pressure_forecast_sa2.parquet")
    if preds_path is None:
        preds_path = next((p for p in latest_candidates if p.exists()), latest_candidates[-1])

    preds = pd.read_parquet(preds_path).copy()
    preds["month"] = to_month(preds["month"])
    preds["sa2_code"] = preds["sa2_code"].astype(str)
    if "rent_jump_threshold" in preds.columns:
        preds = preds[np.isclose(preds["rent_jump_threshold"].astype(float), float(threshold))].copy()
        if preds.empty:
            raise SystemExit(f"No predictions found for threshold={threshold:.3f} in {preds_path}.")
    else:
        preds["rent_jump_threshold"] = float(threshold)

    # Price clusters: prefer stored values, otherwise derive from latest house price snapshots
    if "price_cluster" in preds.columns:
        preds["price_cluster"] = preds["price_cluster"].fillna(-1).astype(int)
    else:
        preds["price_cluster"] = -1

    hp_monthly_path = Path(STAGE_DIR / "house_prices_sa2_monthly.parquet")
    hp_snapshot_path = Path(STAGE_DIR / "house_prices_sa2_snapshot.parquet")
    hp_df: pd.DataFrame | None = None
    if hp_monthly_path.exists():
        hp_df = pd.read_parquet(hp_monthly_path).copy()
    elif hp_snapshot_path.exists():
        hp_df = pd.read_parquet(hp_snapshot_path).copy()
    if hp_df is not None and not hp_df.empty:
        if "median_house_price" in hp_df.columns:
            hp_df["sa2_code"] = hp_df["sa2_code"].astype(str)
            hp_df["month"] = to_month(hp_df["month"])
            price_levels = (
                hp_df.groupby("sa2_code")["median_house_price"].mean()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not price_levels.empty:
                q = min(5, price_levels.nunique())
                if q >= 2:
                    cluster_series = pd.qcut(price_levels, q=q, labels=False, duplicates="drop")
                    cluster_map = cluster_series.astype(int).to_dict()
                else:
                    cluster_map = {code: 0 for code in price_levels.index}
                preds.loc[:, "price_cluster"] = preds["sa2_code"].map(cluster_map).fillna(preds["price_cluster"]).astype(int)

    # Optional month window filter (inclusive)
    if start or end:
        start_m = to_month(start) if start else preds["month"].min()
        end_m = to_month(end) if end else preds["month"].max()
        preds = preds[(preds["month"] >= start_m) & (preds["month"] <= end_m)].copy()

    latest_month = preds["month"].max()
    latest = preds.loc[preds["month"] == latest_month].copy()
    latest_suffix = "" if suffix is None else f"_{suffix}"
    latest.sort_values("price_pressure_prob", ascending=False).to_excel(
        OUT_DIR / f"price_pressure_forecast_sa2_latest{latest_suffix}.xlsx", index=False
    )

    # Named version (join SA2 names from ASGS)
    names = _read_sa2_names()   # sa2_code, optional sa2_name
    named = latest.merge(names, on="sa2_code", how="left")
    cols = ["sa2_code"] + (["sa2_name"] if "sa2_name" in named.columns else []) + ["month", "price_pressure_prob"]
    named[cols].to_excel(OUT_DIR / f"price_pressure_forecast_sa2_latest_named{latest_suffix}.xlsx", index=False)
    print(f"Wrote {OUT_DIR / ('price_pressure_forecast_sa2_latest' + latest_suffix + '.xlsx')}")
    print(f"Wrote {OUT_DIR / ('price_pressure_forecast_sa2_latest_named' + latest_suffix + '.xlsx')} ({len(named)} rows for {latest_month:%Y-%m}).")

    # 2) Score predictions where realized outcomes exist
    labels = _build_realized_labels_v2(threshold)
    # Inner join so we only score months where both prediction and actual exist
    joined = preds.merge(labels, on=["sa2_code", "month"], how="inner")
    if joined.empty:
        print("No realized months yet to score. (Run again once the next bond ZIP is ingested.)")
        return

    if "price_cluster" not in joined.columns:
        joined["price_cluster"] = -1

    # Summary metrics per month (over all months with realized labels)
    summary_rows = []
    per_month_groups = list(joined.groupby("month"))

    cluster_calibration_rows = []
    realized_clusters = joined.dropna(subset=["actual_jump"]).copy()
    if not realized_clusters.empty:
        realized_clusters["price_cluster"] = realized_clusters["price_cluster"].fillna(-1).astype(int)
        for (m, cluster), dfc in realized_clusters.groupby(["month", "price_cluster"], sort=True):
            actual = dfc["actual_jump"].astype(int).to_numpy()
            probs = dfc["price_pressure_prob"].astype(float).to_numpy()
            if not len(probs):
                continue
            cluster_calibration_rows.append({
                "month": pd.Timestamp(m),
                "price_cluster": int(cluster),
                "count": int(len(probs)),
                "actual_rate": float(actual.mean()),
                "predicted_mean": float(probs.mean()),
                "calibration_gap": float(probs.mean() - actual.mean()),
                "brier": float(np.mean((probs - actual) ** 2)),
            })

    # Optional house prices snapshot by SA2
    if hp_df is not None:
        rename_cols = {}
        if "allocation_weight_sum" in hp_df.columns:
            rename_cols["allocation_weight_sum"] = "price_allocation_weight_sum"
        if "n_suburbs" in hp_df.columns:
            rename_cols["n_suburbs"] = "price_suburb_count"
        if rename_cols:
            hp_df = hp_df.rename(columns=rename_cols)
        keep_cols = [c for c in ["sa2_code", "month", "median_house_price", "price_allocation_weight_sum", "price_suburb_count"] if c in hp_df.columns]
        hp_df = hp_df[keep_cols].copy()
        hp_df["sa2_code"] = hp_df["sa2_code"].astype(str)
        hp_df["month"] = to_month(hp_df["month"])  # normalize
    K_LIST = [10, 20, 50, 100]
    for m, dfm in per_month_groups:
        y = dfm["actual_jump"].astype(int).to_numpy()
        p = dfm["price_pressure_prob"].astype(float).to_numpy()

        thr = 0.5
        yhat = (p >= thr).astype(int)

        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())

        # Baselines: seasonal naive (t-12) and LOCF (t-1)
        b_seasonal_mask = dfm["actual_jump_prev_year"].notna().to_numpy()
        b_seasonal = dfm.loc[b_seasonal_mask, "actual_jump_prev_year"].astype(int).to_numpy()
        y_for_seasonal = y[b_seasonal_mask]

        b_locf_mask = dfm["actual_jump_prev"].notna().to_numpy()
        b_locf = dfm.loc[b_locf_mask, "actual_jump_prev"].astype(int).to_numpy()
        y_for_locf = y[b_locf_mask]

        # Error metrics
        mae = float(np.mean(np.abs(p - y)))
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))  # sqrt(Brier)
        # sMAPE with epsilon to avoid 0/0 when y=p=0
        eps = 1e-12
        smape = float(np.mean(2.0 * np.abs(p - y) / (np.abs(p) + np.abs(y) + eps)))

        # WA-weighted errors (by stock_bonds)
        w = dfm.get("stock_bonds", pd.Series(1.0, index=dfm.index)).astype(float).to_numpy()
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.where(w < 0, 0.0, w)
        w_sum = float(w.sum()) if float(w.sum()) > 0 else 1.0
        w_mae = float(np.sum(np.abs(p - y) * w) / w_sum)
        w_rmse = float(np.sqrt(np.sum(((p - y) ** 2) * w) / w_sum))

        # Relative to seasonal naive: MASE = MAE_model / MAE_naive
        if len(y_for_seasonal) > 0:
            mae_seasonal = float(np.mean(np.abs(b_seasonal - y_for_seasonal)))
            mase_seasonal = float(mae / mae_seasonal) if mae_seasonal > 0 else np.nan
            # Brier Skill Score vs seasonal (1 - BS_model/BS_baseline)
            bs_model_seas = _brier(y_for_seasonal, p[b_seasonal_mask])
            bs_baseline_seas = _brier(y_for_seasonal, b_seasonal.astype(float))
            bss_seasonal = float(1.0 - (bs_model_seas / bs_baseline_seas)) if bs_baseline_seas > 0 else np.nan
        else:
            mae_seasonal = np.nan
            mase_seasonal = np.nan
            bss_seasonal = np.nan

        # Relative to LOCF
        if len(y_for_locf) > 0:
            mae_locf = float(np.mean(np.abs(b_locf - y_for_locf)))
            mase_locf = float(mae / mae_locf) if mae_locf > 0 else np.nan
            bs_model_locf = _brier(y_for_locf, p[b_locf_mask])
            bs_baseline_locf = _brier(y_for_locf, b_locf.astype(float))
            bss_locf = float(1.0 - (bs_model_locf / bs_baseline_locf)) if bs_baseline_locf > 0 else np.nan
        else:
            mae_locf = np.nan
            mase_locf = np.nan
            bss_locf = np.nan


        # Best threshold by F1 on this month (evaluate on unique prob cutpoints)
        try:
            cand = np.unique(p)
            # add a small grid if too few candidates
            if cand.size < 5:
                cand = np.unique(np.concatenate([cand, np.linspace(0.1, 0.9, 17)]))
            best_thr = 0.5
            best_f1 = -1.0
            best_prec = np.nan
            best_rec = np.nan
            best_acc = np.nan
            for thr_ in cand:
                yhat_ = (p >= thr_).astype(int)
                tp_ = int(((yhat_ == 1) & (y == 1)).sum())
                fp_ = int(((yhat_ == 1) & (y == 0)).sum())
                fn_ = int(((yhat_ == 0) & (y == 1)).sum())
                tn_ = int(((yhat_ == 0) & (y == 0)).sum())
                prec_ = (tp_ / (tp_ + fp_)) if (tp_ + fp_) else np.nan
                rec_ = (tp_ / (tp_ + fn_)) if (tp_ + fn_) else np.nan
                if np.isnan(prec_) or np.isnan(rec_) or (prec_ + rec_ == 0):
                    f1_ = 0.0
                else:
                    f1_ = 2 * prec_ * rec_ / (prec_ + rec_)
                if f1_ > best_f1:
                    best_f1 = f1_
                    best_thr = float(thr_)
                    best_prec = prec_
                    best_rec = rec_
                    best_acc = ((tp_ + tn_) / len(y)) if len(y) else np.nan
        except Exception:
            best_thr = 0.5
            best_f1 = np.nan
            best_prec = np.nan
            best_rec = np.nan
            best_acc = np.nan

        # Calibration extras
        ece = _ece(y, p, bins=10)
        rel, res, unc = _brier_decompose(y, p, bins=10)

        # House-price correlation (Spearman via rank correlation), if available
        hp_corr = float("nan"); hp_n = 0
        if hp_df is not None and not hp_df.empty:
            hp_latest = _latest_hp_on_or_before(hp_df, m)
            d2 = dfm.merge(hp_latest[["sa2_code", "median_house_price"]], on="sa2_code", how="left")
            d2 = d2.dropna(subset=["median_house_price"]).copy()
            if len(d2) >= 10:
                hp_n = int(len(d2))
                hp_corr = float(d2["price_pressure_prob"].rank().corr(d2["median_house_price"].rank()))

        # Precision@K and lift@K
        base_rate = float(y.mean())
        topk = dfm.sort_values("price_pressure_prob", ascending=False).reset_index(drop=True)
        prec_at_k = {}
        lift_at_k = {}
        pos_at_k = {}
        for K in K_LIST:
            if len(topk) >= K:
                pos_k = int(topk.iloc[:K]["actual_jump"].astype(bool).sum())
                prec_k = float(pos_k / K)
                lift_k = float(prec_k / base_rate) if base_rate > 0 else np.nan
                prec_at_k[f"precision_at_{K}"] = prec_k
                lift_at_k[f"lift_at_{K}"] = lift_k
                pos_at_k[f"positives_in_top_{K}"] = pos_k

        row = {
            "month": m,
            "n_sa2": len(dfm),
            "base_rate": base_rate,
            "auc": _roc_auc(y, p),
            "brier": _brier(y, p),
            "log_loss": _log_loss(y, p),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision_at_0.5": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall_at_0.5": tp / (tp + fn) if (tp + fn) else np.nan,
            "accuracy_at_0.5": (tp + tn) / len(dfm) if len(dfm) else np.nan,
            # Additional metrics
            "mae": mae,
            "rmse": rmse,
            "smape": smape,
            "w_mae": w_mae,
            "w_rmse": w_rmse,
            "mae_seasonal_naive": mae_seasonal,
            "mae_locf_naive": mae_locf,
            "mase_vs_seasonal": mase_seasonal,
            "mase_vs_locf": mase_locf,
            "bss_vs_seasonal": bss_seasonal,
            "bss_vs_locf": bss_locf,
            # Best-threshold diagnostics for operations
            "best_thr_f1": best_thr,
            "best_f1": best_f1,
            "best_precision": best_prec,
            "best_recall": best_rec,
            "best_accuracy": best_acc,
            # Calibration extras
            "ece": ece,
            "brier_reliability": rel,
            "brier_resolution": res,
            "brier_uncertainty": unc,
            # House prices
            "hp_corr_spearman": hp_corr,
            "hp_n": hp_n,
        }
        # Merge P@K/lift counters
        row.update(prec_at_k); row.update(lift_at_k); row.update(pos_at_k)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("month")
    summary_path = EVAL_DIR / "forecast_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    if cluster_calibration_rows:
        cluster_df = pd.DataFrame(cluster_calibration_rows)
        if pd.api.types.is_datetime64_any_dtype(cluster_df["month"]):
            cluster_df["month"] = cluster_df["month"].dt.to_period("M").astype(str)
        cluster_df.sort_values(["month", "price_cluster"], inplace=True)
        cluster_csv = EVAL_DIR / "forecast_calibration_clusters.csv"
        cluster_df.to_csv(cluster_csv, index=False)
        print(f"Wrote {cluster_csv}")

    # 3) Detailed & calibration for each scored month
    for m, dfm in per_month_groups:
        detail = dfm.copy()
        det_path = EVAL_DIR / f"forecast_eval_details_{m:%Y-%m}.csv"
        detail.to_csv(det_path, index=False)
        print(f"Wrote {det_path}")

        cal = _calibration_bins(
            detail["actual_jump"].astype(int).to_numpy(),
            detail["price_pressure_prob"].astype(float).to_numpy(),
            bins=10,
        )
        cal_path = EVAL_DIR / f"forecast_calibration_{m:%Y-%m}.csv"
        cal.to_csv(cal_path, index=False)
        print(f"Wrote {cal_path}")

        # Calibration plot per month
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(cal["pred"], cal["actual"], marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("Predicted probability (bin mean)")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration — {m:%Y-%m}")
        fig.tight_layout()
        png = FIG_DIR / f"forecast_calibration_{m:%Y-%m}.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
        print(f"Wrote {png}")

        # PR curve per month (if sklearn available)
        ap, rec, prec = _prc_metrics(detail["actual_jump"].astype(int).to_numpy(),
                                     detail["price_pressure_prob"].astype(float).to_numpy())
        if rec.size and prec.size:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.step(rec, prec, where='post')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'Precision-Recall — {m:%Y-%m} (AP={ap:.3f})')
            ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])
            pr_png = FIG_DIR / f"forecast_pr_curve_{m:%Y-%m}.png"
            fig2.tight_layout(); fig2.savefig(pr_png, dpi=160); plt.close(fig2)
            print(f"Wrote {pr_png}")

        # Optional: scatter rent-risk vs house price (if snapshot exists)
        if hp_df is not None and not hp_df.empty:
            hp_latest = _latest_hp_on_or_before(hp_df, m)
            d3 = detail.merge(hp_latest[["sa2_code", "median_house_price"]], on="sa2_code", how="left").dropna(subset=["median_house_price"]).copy()
            if len(d3) >= 10:
                fig3, ax3 = plt.subplots(figsize=(6,5))
                ax3.scatter(d3["median_house_price"], d3["price_pressure_prob"], s=10, alpha=0.6)
                ax3.set_xlabel("Median house price (latest on/before month)")
                ax3.set_ylabel("Rent rise probability")
                ax3.set_title(f"Rent risk vs house price — {m:%Y-%m}")
                fig3.tight_layout()
                scat_png = FIG_DIR / f"rent_vs_houseprice_{m:%Y-%m}.png"
                fig3.savefig(scat_png, dpi=160); plt.close(fig3)
                print(f"Wrote {scat_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate forecast accuracy; optionally filter months.")
    parser.add_argument("--threshold", type=float, default=RENT_GROWTH_THRESHOLD,
                        help=f"Rent growth threshold (default {RENT_GROWTH_THRESHOLD})")
    parser.add_argument("--start", type=str, default=None, help="Start month YYYY-MM (inclusive)")
    parser.add_argument("--end", type=str, default=None, help="End month YYYY-MM (inclusive)")
    args = parser.parse_args()
    main(threshold=args.threshold, start=args.start, end=args.end)
