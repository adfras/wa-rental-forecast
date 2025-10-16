"""
Build a static website under docs/ with a monthly time slider map:
 - docs/sa2_wa_simplified.geojson
 - docs/data/YYYY-MM.json   (per-month: prob + actual inputs)
 - docs/summary.json        (AUC, Brier per month with actuals)
 - docs/months.json         (available forecast months)
 - docs/index.html          (Leaflet map with time slider + dynamic tooltips)
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd

from src.common.spatial import detect_sa2_fields, find_sa2_layer, simplify_in_meters
from src.config import STAGE_DIR, ASGS_SA2_GPKG, RENT_GROWTH_THRESHOLD
from src.features.dates import to_month

SITE_DIR = Path("docs")
DATA_DIR = SITE_DIR / "data"

THRESHOLD_SPECS = [
    {
        "id": "thr0p010",
        "label": "Rent rise ≥ 1%",
        "event_threshold": 0.01,
        "history_candidates": [
            "price_pressure_forecast_sa2_history_thr0p010.parquet",
            "price_pressure_forecast_sa2_history_thr0p01.parquet",
        ],
        "eval_candidates": [
            Path("outputs/evaluations/multi_threshold/thr0p010/forecast_eval_summary_thr0p010.csv"),
        ],
    },
    {
        "id": "thr0p020",
        "label": "Rent rise ≥ 2%",
        "event_threshold": 0.02,
        "history_candidates": [
            "price_pressure_forecast_sa2_history_thr0p020.parquet",
            "price_pressure_forecast_sa2_history.parquet",
        ],
        "eval_candidates": [
            Path("outputs/evaluations/multi_threshold/thr0p020/forecast_eval_summary_thr0p020.csv"),
            Path("outputs/evaluations/forecast_eval_summary.csv"),
        ],
        "default": True,
    },
    {
        "id": "thr0p030",
        "label": "Rent rise ≥ 3%",
        "event_threshold": 0.03,
        "history_candidates": [
            "price_pressure_forecast_sa2_history_thr0p030.parquet",
            "price_pressure_forecast_sa2_history_thr0p03.parquet",
        ],
        "eval_candidates": [
            Path("outputs/evaluations/multi_threshold/thr0p030/forecast_eval_summary_thr0p030.csv"),
        ],
    },
]

DEFAULT_THRESHOLD_ID = next((spec["id"] for spec in THRESHOLD_SPECS if spec.get("default")), THRESHOLD_SPECS[0]["id"])

# ----------------- helpers -----------------

def _write_json(path: Path, obj):
    """Write JSON with UTF-8, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def _read_docs_predictions(base_dir: Path) -> pd.DataFrame:
    """Parse docs data JSON files under ``base_dir`` into a predictions DataFrame."""
    if not base_dir.exists():
        return pd.DataFrame(columns=["sa2_code", "month", "price_pressure_prob"]).astype({"sa2_code": str})
    rows = []
    for p in sorted(base_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")):
        month_str = p.stem
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        for sa2_code, rec in obj.items():
            pval = rec.get("p", None)
            lval = rec.get("l", None)
            uval = rec.get("u", None)
            rows.append({
                "sa2_code": str(sa2_code),
                "month": pd.to_datetime(month_str).to_period("M").to_timestamp(),
                "price_pressure_prob": (None if pval is None else float(pval)),
                "prob_p05": (None if lval is None else float(lval)),
                "prob_p95": (None if uval is None else float(uval)),
            })
    if not rows:
        return pd.DataFrame(columns=["sa2_code", "month", "price_pressure_prob"]).astype({"sa2_code": str})
    df = pd.DataFrame(rows)
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df

# ----------------- build geometry once -----------------

def build_geometry_geojson() -> dict:
    layer = find_sa2_layer(ASGS_SA2_GPKG)
    gdf = gpd.read_file(ASGS_SA2_GPKG, layer=layer)
    code_col, name_col = detect_sa2_fields(gdf)

    keep_cols = [code_col, "geometry"]
    if name_col:
        keep_cols.insert(1, name_col)
    gdf = gdf[keep_cols].copy()
    gdf.rename(columns={code_col: "sa2_code"}, inplace=True)
    if name_col:
        gdf.rename(columns={name_col: "sa2_name"}, inplace=True)

    # Keep only WA SA2s (codes start with '5'); ensure str type
    gdf["sa2_code"] = gdf["sa2_code"].astype(str)
    gdf = gdf[gdf["sa2_code"].str.startswith("5")].copy()

    # Simplify a bit for web perf
    gdf = simplify_in_meters(gdf, tol_m=200)  # increase to 300–500 for smaller files

    # Ensure non-geom fields are strings for GeoJSON
    for c in gdf.columns:
        if c != "geometry":
            gdf[c] = gdf[c].astype(str)

    geojson = json.loads(gdf.to_json())
    _write_json(SITE_DIR / "sa2_wa_simplified.geojson", geojson)
    return geojson

# ----------------- build month data & summary -----------------
def _format_threshold_label(value: float) -> str:
    pct = value * 100
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}%"
    return f"{pct:.1f}%"


def _gather_history_paths(spec: dict) -> list[Path]:
    paths: list[Path] = []
    for candidate in spec.get("history_candidates", []):
        path = STAGE_DIR / candidate
        if path.exists():
            paths.append(path)
    return paths


def _resolve_eval_summary_path(spec: dict) -> Path | None:
    for candidate in spec.get("eval_candidates", []):
        if candidate.exists():
            return candidate
    return None


def _load_best_thresholds(summary_path: Path | None) -> dict[str, float]:
    if summary_path is None:
        return {}
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return {}
    if "month" not in df.columns or "best_thr_f1" not in df.columns:
        return {}
    mapping: dict[str, float] = {}
    months = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    for month_str, thr in zip(months, df["best_thr_f1"].to_numpy()):
        try:
            mapping[str(month_str)] = float(thr)
        except Exception:
            continue
    return mapping


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(p, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    uniq, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        sum_ranks = np.bincount(inv, weights=ranks, minlength=len(uniq))
        mean_ranks = sum_ranks / cnt
        ranks = mean_ranks[inv]
    n1 = float(y.sum())
    n0 = float(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    return float(np.mean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _best_f1_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return the (threshold, precision, recall, accuracy) that maximises F1."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)

    mask = np.isfinite(p)
    p = p[mask]
    y = y[mask]

    # Need at least one observation with an actual label to compute anything sensible
    if p.size == 0 or y.size == 0 or (y.sum() == 0 and (1 - y).sum() == 0):
        return 0.5, float("nan"), float("nan"), float("nan")

    candidates = np.unique(p)
    if candidates.size < 5:
        candidates = np.unique(np.concatenate([candidates, np.linspace(0.1, 0.9, 17)]))

    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = float("nan")
    best_recall = float("nan")
    best_accuracy = float("nan")

    for thr in candidates:
        if not np.isfinite(thr):
            continue
        yhat = (p >= thr).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())

        precision = (tp / (tp + fp)) if (tp + fp) else float("nan")
        recall = (tp / (tp + fn)) if (tp + fn) else float("nan")

        if np.isnan(precision) or np.isnan(recall) or (precision + recall == 0):
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Prefer the lowest threshold on ties so recall does not collapse
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and thr < best_threshold):
            best_f1 = f1
            best_threshold = float(thr)
            best_precision = precision
            best_recall = recall
            best_accuracy = ((tp + tn) / len(y)) if len(y) else float("nan")

    return best_threshold, best_precision, best_recall, best_accuracy


def build_threshold_dataset(spec: dict, sa2_panel: pd.DataFrame, docs_prev: pd.DataFrame | None) -> dict | None:
    history_paths = _gather_history_paths(spec)
    preds_frames: list[pd.DataFrame] = []
    if docs_prev is not None and not docs_prev.empty:
        preds_frames.append(docs_prev.copy())

    for path in history_paths:
        try:
            df_p = pd.read_parquet(path).copy()
        except Exception as exc:
            print(f"[warn] Failed to read {path} for {spec['id']}: {exc}")
            continue
        if df_p.empty:
            continue
        preds_frames.append(df_p)

    if not preds_frames:
        print(f"[warn] No forecast history for {spec['id']} — skipping threshold")
        return None

    preds = pd.concat(preds_frames, ignore_index=True)
    preds["month"] = to_month(preds["month"])
    preds["sa2_code"] = preds["sa2_code"].astype(str)
    preds = preds.sort_values(["sa2_code", "month"])
    preds = preds.drop_duplicates(subset=["sa2_code", "month"], keep="last")
    for col in ("prob_p05", "prob_p95"):
        if col not in preds.columns:
            preds[col] = np.nan

    # Actual labels at the requested rent-change threshold
    labels = sa2_panel.loc[sa2_panel["rent_prev"].notna(), ["sa2_code", "month", "rent_return"]].copy()
    labels["actual"] = labels["rent_return"] > spec["event_threshold"]
    joined = preds.merge(labels[["sa2_code", "month", "actual"]], on=["sa2_code", "month"], how="left")

    months = sorted(set(joined["month"].unique().tolist()))
    month_strs = [pd.Timestamp(m).strftime("%Y-%m") for m in months]

    best_thr = _load_best_thresholds(_resolve_eval_summary_path(spec))

    threshold_dir = DATA_DIR / spec["id"]
    threshold_dir.mkdir(parents=True, exist_ok=True)

    payload_cache: dict[str, dict] = {}
    summary: dict[str, dict] = {}

    for m in months:
        dfm = joined[joined["month"] == m]
        m_str = pd.Timestamp(m).strftime("%Y-%m")
        actual_mask = dfm["actual"].notna()

        thr_raw = best_thr.get(m_str)
        thr_m: float | None
        try:
            thr_m = float(thr_raw) if thr_raw is not None else None
        except (TypeError, ValueError):
            thr_m = None

        if (thr_m is None or not np.isfinite(thr_m)) and actual_mask.any():
            p_vals = dfm.loc[actual_mask, "price_pressure_prob"].to_numpy()
            y_vals = dfm.loc[actual_mask, "actual"].astype(int).to_numpy()
            thr_m, _, _, _ = _best_f1_threshold(p_vals, y_vals)
            best_thr[m_str] = thr_m

        if thr_m is None or not np.isfinite(thr_m):
            thr_m = 0.33

        payload: dict[str, dict] = {}
        for _, r in dfm.iterrows():
            p = None if pd.isna(r.get("price_pressure_prob", np.nan)) else float(r["price_pressure_prob"])
            l = None if pd.isna(r.get("prob_p05", np.nan)) else float(r["prob_p05"])
            u = None if pd.isna(r.get("prob_p95", np.nan)) else float(r["prob_p95"])
            y = None if pd.isna(r.get("actual", np.nan)) else int(bool(r["actual"]))
            pred = None if p is None else int(p >= thr_m)
            cls = None
            if y is not None and pred is not None:
                if pred == 1 and y == 1:
                    cls = "TP"
                elif pred == 1 and y == 0:
                    cls = "FP"
                elif pred == 0 and y == 0:
                    cls = "TN"
                else:
                    cls = "FN"
            err = None
            ae = None
            if p is not None and y is not None:
                err = float(p - y)
                ae = abs(err)
            correct = None
            if cls is not None:
                correct = cls in {"TP", "TN"}
            payload[str(r["sa2_code"])] = {
                "p": p,
                "l": l,
                "u": u,
                "y": y,
                "a": pred,
                "pred": pred,
                "correct": correct,
                "cls": cls,
                "err": err,
                "ae": ae,
            }

        payload_cache[m_str] = payload
        _write_json(threshold_dir / f"{m_str}.json", payload)

        if actual_mask.any():
            p = dfm.loc[actual_mask, "price_pressure_prob"].to_numpy()
            y = dfm.loc[actual_mask, "actual"].astype(int).to_numpy()
            yhat_default = (p >= 0.5).astype(int)
            yhat_thr = (p >= thr_m).astype(int)
            tp = int(((yhat_default == 1) & (y == 1)).sum())
            fp = int(((yhat_default == 1) & (y == 0)).sum())
            tn = int(((yhat_default == 0) & (y == 0)).sum())
            fn = int(((yhat_default == 0) & (y == 1)).sum())
            tp_t = int(((yhat_thr == 1) & (y == 1)).sum())
            fp_t = int(((yhat_thr == 1) & (y == 0)).sum())
            tn_t = int(((yhat_thr == 0) & (y == 0)).sum())
            fn_t = int(((yhat_thr == 0) & (y == 1)).sum())

            summary[m_str] = {
                "has_actual": True,
                "event_threshold": float(spec["event_threshold"]),
                "n_sa2": int(actual_mask.sum()),
                "base_rate": float(y.mean()),
                "auc": _roc_auc(y, p),
                "brier": _brier(y, p),
                "log_loss": _log_loss(y, p),
                "precision_at_0_5": (tp / (tp + fp)) if (tp + fp) else float("nan"),
                "recall_at_0_5": (tp / (tp + fn)) if (tp + fn) else float("nan"),
                "accuracy_at_0_5": ((tp + tn) / actual_mask.sum()) if actual_mask.sum() else float("nan"),
                "threshold": float(thr_m),
                "precision_at_thr": (tp_t / (tp_t + fp_t)) if (tp_t + fp_t) else float("nan"),
                "recall_at_thr": (tp_t / (tp_t + fn_t)) if (tp_t + fn_t) else float("nan"),
                "accuracy_at_thr": ((tp_t + tn_t) / actual_mask.sum()) if actual_mask.sum() else float("nan"),
                "tp_thr": int(tp_t),
                "fp_thr": int(fp_t),
                "tn_thr": int(tn_t),
                "fn_thr": int(fn_t),
            }
        else:
            summary[m_str] = {
                "has_actual": False,
                "event_threshold": float(spec["event_threshold"]),
                "n_sa2": int(len(dfm)),
                "base_rate": None,
                "auc": None,
                "brier": None,
                "log_loss": None,
                "precision_at_0_5": None,
                "recall_at_0_5": None,
                "accuracy_at_0_5": None,
                "threshold": float(thr_m),
                "precision_at_thr": None,
                "recall_at_thr": None,
                "accuracy_at_thr": None,
                "tp_thr": None,
                "fp_thr": None,
                "tn_thr": None,
                "fn_thr": None,
            }

    _write_json(threshold_dir / "months.json", month_strs)
    _write_json(threshold_dir / "summary.json", summary)

    if spec["id"] == DEFAULT_THRESHOLD_ID:
        _write_json(DATA_DIR / "months.json", month_strs)
        _write_json(DATA_DIR / "summary.json", summary)
        for m_str, payload in payload_cache.items():
            _write_json(DATA_DIR / f"{m_str}.json", payload)

    label = spec.get("label")
    if not label:
        label = f"Rent rise ≥ {_format_threshold_label(spec['event_threshold'])}"
    return {
        "id": spec["id"],
        "label": label,
        "event_threshold": float(spec["event_threshold"]),
        "default": bool(spec.get("default", False)),
        "months": month_strs,
    }


def build_threshold_datasets(existing_docs: dict[str | None, pd.DataFrame]) -> list[dict]:
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    sa2["month"] = to_month(sa2["month"])
    sa2["sa2_code"] = sa2["sa2_code"].astype(str)
    sa2 = sa2.sort_values(["sa2_code", "month"])
    sa2["rent_prev"] = sa2.groupby("sa2_code")["median_rent"].shift(1)
    sa2["rent_return"] = np.where(
        sa2["rent_prev"].notna() & (sa2["rent_prev"] != 0),
        sa2["median_rent"] / sa2["rent_prev"] - 1.0,
        np.nan,
    )

    infos: list[dict] = []
    for spec in THRESHOLD_SPECS:
        docs_prev = existing_docs.get(spec["id"])
        if spec["id"] == DEFAULT_THRESHOLD_ID:
            root_docs = existing_docs.get(None)
            if root_docs is not None and not root_docs.empty:
                if docs_prev is not None and not docs_prev.empty:
                    docs_prev = pd.concat([docs_prev, root_docs], ignore_index=True)
                else:
                    docs_prev = root_docs
        info = build_threshold_dataset(spec, sa2, docs_prev)
        if info is not None:
            infos.append(info)

    thresholds_json = [
        {
            "id": info["id"],
            "label": info.get("label"),
            "event_threshold": info["event_threshold"],
            "default": info.get("default", False),
        }
        for info in infos
    ]
    _write_json(DATA_DIR / "thresholds.json", thresholds_json)
    return infos

# ----------------- write index.html -----------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>WA Rental Forecast — Accuracy Overview</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<style>
  html, body, #map { height: 100%; margin: 0; }
  body { font-family: system-ui, sans-serif; color: #1f1f1f; }
  .panel {
    position: absolute; top: 12px; left: 12px; z-index: 1000;
    background: rgba(255,255,255,0.96); padding: 12px 14px; border-radius: 10px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.16); max-width: 380px;
  }
  .panel h1 { font-size: 16px; margin: 0 0 8px; font-weight: 600; }
  .row { display:flex; gap: 8px; align-items:center; margin: 8px 0; flex-wrap: wrap; }
  label { font-size: 12px; color: #333; }
  select, input[type=range], button { font-size: 12px; }
  input[type=range] { flex: 1 1 160px; }
  #monthLabel { min-width: 56px; text-align: center; padding:2px 8px; border-radius:10px; background:#f0f0f0; }
  .legend { margin-top: 10px; font-size: 12px; line-height: 1.4; }
  .legend .grad { height: 10px; border:1px solid #999; margin:4px 0; }
  .swatch { display:inline-block; width:18px; height:12px; border:1px solid #777; margin-right:6px; vertical-align:middle; }
  .footer {
    position: absolute; bottom: 12px; left: 12px; z-index: 1000;
    background: rgba(255,255,255,0.92); padding: 8px 10px; border-radius: 8px;
    font-size: 12px; box-shadow: 0 1px 10px rgba(0,0,0,0.12); max-width: 520px;
  }
  .footer strong { font-weight: 600; }
  button { border:1px solid #bbb; background:#f7f7f7; border-radius:4px; padding:2px 6px; cursor:pointer; }
  button:hover { background:#ececec; }
  #playBtn { width:26px; height:22px; display:flex; align-items:center; justify-content:center; }
</style>
</head>
<body>
<div id="map"></div>

<div class="panel">
  <h1>WA Rental Forecast — Accuracy</h1>
  <div class="row">
    <label for="thresholdSelect">Event threshold</label>
    <select id="thresholdSelect"></select>
  </div>
  <div class="row">
    <label for="metric">View</label>
    <select id="metric">
      <option value="forecast">Forecast probability</option>
      <option value="actual">Actual outcome</option>
      <option value="correctness">Model correctness</option>
    </select>
  </div>
  <div class="row">
    <label for="monthRange">Month</label>
    <input id="monthRange" type="range" min="0" max="0" step="1" value="0"/>
    <span id="monthLabel">—</span>
    <button id="playBtn" title="Play animation">▶</button>
  </div>
  <div class="row">
    <label><input type="checkbox" id="alertsOnly"/> Highlight alerts</label>
  </div>
  <div id="legend" class="legend"></div>
</div>

<div class="footer" id="metrics">Loading…</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script>
const THRESHOLDS = __THRESHOLDS_JSON__;
const DEFAULT_THRESHOLD = __DEFAULT_THRESHOLD__;
let currentThreshold = THRESHOLDS.length ? (THRESHOLDS.find(t => t.default) || THRESHOLDS[0]) : DEFAULT_THRESHOLD;

let map, layer;
let months = [];
let monthIdx = 0;
let monthData = {};
let summary = {};
let latestActualIdx = null;

let thresholdSelect, metricSelect, monthRange, alertsOnlyCb, playBtn, metricsEl;
let playHandle = null;
let currentMonthHasActual = false;

function hasActualValues(data) {
  if (!data) return false;
  return Object.values(data).some(rec => rec && rec.y !== null && rec.y !== undefined);
}
function eventThresholdLabel(value) {
  const thr = Number.isFinite(value) ? value : (currentThreshold ? currentThreshold.event_threshold : DEFAULT_THRESHOLD.event_threshold);
  const pct = thr * 100;
  return (Math.abs(pct - Math.round(pct)) < 1e-6 ? Math.round(pct).toString() : pct.toFixed(1)) + '%';
}
function dataUrl(filename) {
  const base = currentThreshold && currentThreshold.id ? currentThreshold.id : DEFAULT_THRESHOLD.id;
  return `data/${base}/${filename}`;
}
function fmtPct(val, digits = 1) {
  return Number.isFinite(val) ? (val * 100).toFixed(digits) + '%' : '—';
}
function fmtPct0(val) { return fmtPct(val, 0); }
function fmtFloat(val, digits = 3) { return Number.isFinite(val) ? val.toFixed(digits) : '—'; }
function fmtCount(val) { return Number.isFinite(val) ? String(Math.round(val)) : '—'; }
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function lerpColor(a,b,t){
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ar = (pa >> 16) & 0xff, ag = (pa >> 8) & 0xff, ab = pa & 0xff;
  const br = (pb >> 16) & 0xff, bg = (pb >> 8) & 0xff, bb = pb & 0xff;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const blue = Math.round(ab + (bb - ab) * t);
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + blue).toString(16).slice(1);
}
function colorProb(p){ if(p==null || isNaN(p)) return '#d9d9d9';
  const stops=[[0.0,'#ffffcc'],[0.2,'#ffeda0'],[0.4,'#feb24c'],[0.6,'#fd8d3c'],[0.8,'#f03b20'],[1.0,'#bd0026']];
  for (let i=1;i<stops.length;i++){
    if (p <= stops[i][0]){
      const t = (p - stops[i-1][0]) / (stops[i][0] - stops[i-1][0]);
      return lerpColor(stops[i-1][1], stops[i][1], t);
    }
  }
  return stops[stops.length-1][1];
}
function colorActual(y){ if(y==null) return '#d9d9d9'; return y ? '#d73027' : '#1a9850'; }
function colorCorrect(isCorrect){
  if(isCorrect == null) return '#d9d9d9';
  return isCorrect ? '#31a354' : '#ef3b2c';
}

function setLegend(kind){
  const el = document.getElementById('legend');
  if(!el) return;
  const label = eventThresholdLabel();
  if(kind === 'forecast'){
    el.innerHTML = `<div><b>Forecast probability</b> (Pr[Δrent &gt; ${label}])</div>
      <div class="grad" style="background:linear-gradient(to right,#ffffcc,#ffeda0,#feb24c,#fd8d3c,#f03b20,#bd0026)"></div>
      <div style="display:flex;justify-content:space-between;"><span>0</span><span>1</span></div>`;
  } else if(kind === 'actual'){
    el.innerHTML = `<div><b>Actual outcome</b> (rise &gt; ${label})</div>
      <div style="display:flex;flex-direction:column;gap:4px;margin-top:4px;">
        <span><span class="swatch" style="background:#d73027"></span> Rise</span>
        <span><span class="swatch" style="background:#1a9850"></span> No rise</span>
        <span><span class="swatch" style="background:#d9d9d9"></span> Not yet known</span>
      </div>`;
  } else {
    el.innerHTML = `<div><b>Model correctness</b> (best operating threshold)</div>
      <div style="display:flex;flex-direction:column;gap:4px;margin-top:4px;">
        <span><span class="swatch" style="background:#31a354"></span> Model was correct</span>
        <span><span class="swatch" style="background:#ef3b2c"></span> Model was incorrect</span>
        <span><span class="swatch" style="background:#d9d9d9"></span> Not yet known</span>
      </div>`;
  }
}

function styleFeature(feat){
  const code = feat.properties.sa2_code;
  const rec = monthData ? monthData[code] : null;
  const metric = metricSelect ? metricSelect.value : 'forecast';
  const alertsOnly = alertsOnlyCb ? alertsOnlyCb.checked : false;
  let fill = '#d9d9d9';
  if(rec){
    if(metric === 'forecast') fill = colorProb(rec.p);
    else if(metric === 'actual') fill = colorActual(rec.y);
    else if(metric === 'correctness') fill = colorCorrect(rec.correct);
  }
  if(alertsOnly && (!rec || rec.pred !== 1)){
    return { fillColor: fill, color: '#666', weight: 0.0, fillOpacity: 0.0 };
  }
  return { fillColor: fill, color: '#666', weight: 0.4, fillOpacity: 0.85 };
}

function tooltipHTML(feat){
  const code = feat.properties.sa2_code;
  const name = feat.properties.sa2_name || '';
  const rec = monthData ? monthData[code] : null;
  const p = rec ? rec.p : null;
  const y = rec ? rec.y : null;
  const pred = rec ? rec.pred : null;
  const correct = rec ? rec.correct : null;
  const l = rec ? rec.l : null;
  const u = rec ? rec.u : null;
  const ci = (l==null || u==null) ? '—' : `${fmtPct(l)} – ${fmtPct(u)}`;
  const actualLabel = y == null ? 'Not yet known' : (y ? 'Yes' : 'No');
  const predLabel = pred == null ? 'Not yet run' : (pred ? 'Raise' : 'No raise');
  const correctLabel = correct == null ? 'Not yet known' : (correct ? 'Yes ✔' : 'No ✖');
  const label = eventThresholdLabel();
  return `<p class="tooltip-line"><b>${name || code}</b></p>
          <p class="tooltip-line">SA2: ${code}</p>
          <p class="tooltip-line">Forecast: ${fmtPct(p)}</p>
          <p class="tooltip-line">90% CI: ${ci}</p>
          <p class="tooltip-line">Actual rise &gt; ${label}: ${actualLabel}</p>
          <p class="tooltip-line">Model prediction: ${predLabel}</p>
          <p class="tooltip-line">Model correct: ${correctLabel}</p>`;
}

function updateFooter(info, month){
  if(!metricsEl) metricsEl = document.getElementById('metrics');
  if(!metricsEl){ return; }
  const label = eventThresholdLabel(info && info.event_threshold);
  if(!info || !info.has_actual){
    const count = info && Number.isFinite(info.n_sa2) ? info.n_sa2 : '—';
    metricsEl.innerHTML = `<strong>${month}</strong>: Actual outcomes are not yet available. Forecasts cover ${count} suburbs (rent-rise threshold ${label}).`;
    return;
  }
  const acc = fmtPct0(info.accuracy_at_thr);
  const prec = fmtPct0(info.precision_at_thr);
  const rec = fmtPct0(info.recall_at_thr);
  const base = fmtPct0(info.base_rate);
  const thr = fmtFloat(info.threshold, 3);
  metricsEl.innerHTML = `<strong>${month}</strong>: Model was correct for ${acc} of suburbs (precision ${prec}, recall ${rec}). Base rent-rise rate ${base}; rent threshold ${label}. Best-F1 cutoff ${thr}.`;
}

async function loadMonth(idx){
  if(!months.length){
    monthIdx = 0;
    if(monthRange) monthRange.value = '0';
    const labelEl = document.getElementById('monthLabel');
    if(labelEl) labelEl.textContent = '—';
    monthData = {};
    currentMonthHasActual = false;
    updateFooter(null, '—');
    if(layer) layer.setStyle(styleFeature);
    setLegend(metricSelect ? metricSelect.value : 'forecast');
    return;
  }
  monthIdx = Math.max(0, Math.min(idx, months.length-1));
  const month = months[monthIdx];
  if(monthRange) monthRange.value = String(monthIdx);
  const labelEl = document.getElementById('monthLabel');
  if(labelEl) labelEl.textContent = month;
  try {
    const monthUrl = dataUrl(`${month}.json`);
    const res = await fetch(`${monthUrl}?ts=` + Date.now());
    monthData = await res.json();
  } catch (err) {
    monthData = {};
  }
  currentMonthHasActual = hasActualValues(monthData);
  const info = summary[month] || null;
  updateFooter(info, month);
  if(layer){
    layer.setStyle(styleFeature);
    layer.eachLayer(ln => {
      const tt = ln.getTooltip && ln.getTooltip();
      if(tt && ln.isTooltipOpen && ln.isTooltipOpen()){
        tt.setContent(tooltipHTML(ln.feature));
      }
    });
  }
  const kind = metricSelect ? metricSelect.value : 'forecast';
  if((kind === 'actual' || kind === 'correctness') && !currentMonthHasActual && latestActualIdx !== null){
    metricSelect.value = 'forecast';
    setLegend('forecast');
  } else {
    setLegend(kind);
  }
}

function stopPlayback(){
  if(playHandle){
    clearInterval(playHandle);
    playHandle = null;
    if(playBtn) playBtn.textContent = '▶';
  }
}

async function loadThreshold(thresholdId, preferMonth = null){
  stopPlayback();
  const next = THRESHOLDS.find(t => t.id === thresholdId);
  currentThreshold = next || DEFAULT_THRESHOLD;
  if(thresholdSelect) thresholdSelect.value = currentThreshold.id;

  try {
    const monthsRes = await fetch(`${dataUrl('months.json')}?ts=${Date.now()}`);
    const payload = await monthsRes.json();
    months = Array.isArray(payload) ? payload : [];
  } catch (err) {
    months = [];
  }

  try {
    const summaryRes = await fetch(`${dataUrl('summary.json')}?ts=${Date.now()}`);
    summary = await summaryRes.json();
  } catch (err) {
    summary = {};
  }

  latestActualIdx = null;
  for(let i=0; i<months.length; i++){
    const info = summary[months[i]];
    if(info && info.has_actual) latestActualIdx = i;
  }

  let defaultIdx = months.length ? months.length - 1 : 0;
  if(latestActualIdx !== null) defaultIdx = latestActualIdx;
  if(preferMonth){
    const idx = months.indexOf(preferMonth);
    if(idx >= 0) defaultIdx = idx;
  }

  if(monthRange){
    monthRange.max = months.length ? String(months.length - 1) : '0';
    monthRange.value = months.length ? String(defaultIdx) : '0';
  }

  if(months.length){
    await loadMonth(defaultIdx);
  } else {
    monthIdx = 0;
    if(monthRange) monthRange.value = '0';
    const labelEl = document.getElementById('monthLabel');
    if(labelEl) labelEl.textContent = '—';
    monthData = {};
    currentMonthHasActual = false;
    updateFooter(null, '—');
    if(layer) layer.setStyle(styleFeature);
  }

  if(metricSelect){
    const info = months.length ? summary[months[Math.min(defaultIdx, months.length - 1)]] : null;
    if(info && info.has_actual){
      metricSelect.value = 'correctness';
    } else {
      metricSelect.value = 'forecast';
    }
  }

  setLegend(metricSelect ? metricSelect.value : 'forecast');
  if(layer) layer.setStyle(styleFeature);
}

async function init(){
  map = L.map('map').setView([-31.95, 115.86], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors' }).addTo(map);

  thresholdSelect = document.getElementById('thresholdSelect');
  metricSelect = document.getElementById('metric');
  monthRange = document.getElementById('monthRange');
  alertsOnlyCb = document.getElementById('alertsOnly');
  playBtn = document.getElementById('playBtn');
  metricsEl = document.getElementById('metrics');

  if(thresholdSelect){
    thresholdSelect.innerHTML = '';
    THRESHOLDS.forEach(spec => {
      const opt = document.createElement('option');
      opt.value = spec.id;
      opt.textContent = spec.label || `Rent rise ≥ ${eventThresholdLabel(spec.event_threshold)}`;
      thresholdSelect.appendChild(opt);
    });
    if(!THRESHOLDS.length){
      const opt = document.createElement('option');
      opt.value = DEFAULT_THRESHOLD.id;
      opt.textContent = DEFAULT_THRESHOLD.label || `Rent rise ≥ ${eventThresholdLabel(DEFAULT_THRESHOLD.event_threshold)}`;
      thresholdSelect.appendChild(opt);
    }
    thresholdSelect.value = currentThreshold.id;
    thresholdSelect.disabled = thresholdSelect.options.length <= 1;
    thresholdSelect.addEventListener('change', async e => {
      await loadThreshold(e.target.value, months[monthIdx] || null);
    });
  }

  if(monthRange){
    monthRange.addEventListener('input', e => {
      stopPlayback();
      loadMonth(Number(e.target.value));
    });
  }

  if(metricSelect){
    metricSelect.addEventListener('change', async () => {
      const kind = metricSelect.value;
      if((kind === 'actual' || kind === 'correctness') && !currentMonthHasActual && latestActualIdx !== null){
        if(monthRange) monthRange.value = String(latestActualIdx);
        await loadMonth(latestActualIdx);
        metricSelect.value = kind;
      }
      setLegend(metricSelect.value);
      if(layer){
        layer.setStyle(styleFeature);
      }
    });
  }

  if(alertsOnlyCb){
    alertsOnlyCb.addEventListener('change', () => {
      if(layer) layer.setStyle(styleFeature);
    });
  }

  if(playBtn){
    playBtn.addEventListener('click', () => {
      if(!months.length) return;
      if(playHandle){
        stopPlayback();
      } else {
        playBtn.textContent = '⏸';
        playHandle = setInterval(() => {
          let next = (Number(monthRange.value) + 1) % months.length;
          monthRange.value = String(next);
          loadMonth(next);
        }, 1500);
      }
    });
  }

  const geom = await (await fetch('sa2_wa_simplified.geojson?ts=' + Date.now())).json();
  layer = L.geoJSON(geom, {
    style: styleFeature,
    onEachFeature: (feat, ln) => {
      ln.bindTooltip('', {sticky:true});
      ln.on('tooltipopen', e => e.tooltip.setContent(tooltipHTML(feat)));
      ln.on('mouseover', () => ln.setStyle({weight: 2}));
      ln.on('mouseout', () => ln.setStyle({weight: 0.4}));
    }
  }).addTo(map);
  await loadThreshold(currentThreshold.id);
}

init();
</script>
</body>
</html>
"""

def write_index(threshold_infos: list[dict]):
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    thresholds_json = json.dumps([
        {
            "id": info["id"],
            "label": info.get("label"),
            "event_threshold": info.get("event_threshold", RENT_GROWTH_THRESHOLD),
            "default": bool(info.get("default", False)),
        }
        for info in threshold_infos
    ], ensure_ascii=False)
    if threshold_infos:
        default_info = next((info for info in threshold_infos if info.get("default")), threshold_infos[0])
    else:
        default_info = {
            "id": DEFAULT_THRESHOLD_ID,
            "label": f"Rent rise ≥ {_format_threshold_label(RENT_GROWTH_THRESHOLD)}",
            "event_threshold": RENT_GROWTH_THRESHOLD,
            "default": True,
        }
    default_json = json.dumps({
        "id": default_info.get("id", DEFAULT_THRESHOLD_ID),
        "label": default_info.get("label"),
        "event_threshold": default_info.get("event_threshold", RENT_GROWTH_THRESHOLD),
        "default": True,
    }, ensure_ascii=False)
    index_html = INDEX_HTML.replace("__THRESHOLDS_JSON__", thresholds_json).replace("__DEFAULT_THRESHOLD__", default_json)
    (SITE_DIR / "index.html").write_text(index_html, encoding="utf-8")

# ----------------- main -----------------

def main():
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    existing_docs: dict[str | None, pd.DataFrame] = {spec["id"]: pd.DataFrame() for spec in THRESHOLD_SPECS}
    existing_docs[None] = pd.DataFrame()
    if DATA_DIR.exists():
        existing_docs = {}
        for spec in THRESHOLD_SPECS:
            existing_docs[spec["id"]] = _read_docs_predictions(DATA_DIR / spec["id"])
        existing_docs[None] = _read_docs_predictions(DATA_DIR)
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    build_geometry_geojson()
    threshold_infos = build_threshold_datasets(existing_docs)
    write_index(threshold_infos)
    print(f"Built static site → {SITE_DIR}/ (open docs/index.html)")

if __name__ == "__main__":
    main()
