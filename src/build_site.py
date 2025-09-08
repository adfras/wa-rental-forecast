# src/build_site.py
# Build a static website under docs/ with a monthly time slider map:
# - docs/sa2_wa_simplified.geojson
# - docs/data/YYYY-MM.json   (per-month: prob + actual inputs)
# - docs/summary.json        (AUC, Brier per month with actuals)
# - docs/months.json         (available forecast months)
# - docs/index.html          (Leaflet map with time slider + dynamic tooltips)
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd

# Shapely precision handling (2.x provides set_precision; 1.x needs transform fallback)
try:
    from shapely import set_precision as _set_precision  # Shapely 2.x
    _HAVE_SET_PRECISION = True
except Exception:
    _HAVE_SET_PRECISION = False
    from shapely.ops import transform as _shp_transform  # Shapely 1.x

from src.config import STAGE_DIR, ASGS_SA2_GPKG, RENT_GROWTH_THRESHOLD

SITE_DIR = Path("docs")
DATA_DIR = SITE_DIR / "data"

# ----------------- helpers -----------------

def _to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()

def _find_sa2_layer(path: Path) -> str:
    """Pick the SA2 layer from the ASGS GeoPackage."""
    try:
        from pyogrio import list_layers
        names = [n[0] if isinstance(n, (list, tuple)) else n for n in list_layers(path)]
        for prefer in ("SA2_2021_AUST_GDA2020", "SA2_2021_AUST_GDA94"):
            if prefer in names:
                return prefer
        for n in names:
            if "SA2" in str(n).upper():
                return n
        return names[0]
    except Exception:
        return "SA2_2021_AUST_GDA2020"

def _first_present(cols, cands):
    for c in cands:
        if c in cols:
            return c
    raise ValueError(f"Missing any of {cands}; have: {cols[:12]} ...")

def _write_json(path: Path, obj):
    """Write JSON with UTF-8, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def _read_docs_predictions() -> pd.DataFrame:
    """Parse docs/data/YYYY-MM.json files into a predictions DataFrame.
    Fallback to preserve slider history when parquet history is missing.
    Returns columns: sa2_code (str), month (Timestamp), price_pressure_prob (float).
    """
    if not DATA_DIR.exists():
        return pd.DataFrame(columns=["sa2_code", "month", "price_pressure_prob"]).astype({"sa2_code": str})
    rows = []
    for p in sorted(DATA_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")):
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

def _simplify_in_meters(gdf: gpd.GeoDataFrame, tol_m: float = 200) -> gpd.GeoDataFrame:
    """
    Simplify geometry in meters (EPSG:3577), precision-reduce safely on
    Shapely 2.x or 1.x, then return to WGS84.
    """
    g = gdf.copy()
    # Simplify in meters
    try:
        g = g.to_crs(3577)
        g["geometry"] = g.geometry.simplify(tol_m, preserve_topology=True)
        g = g.to_crs(4326)
    except Exception:
        # Fallback: simplify in degrees (approx)
        g["geometry"] = g.geometry.simplify(tol_m / 111_000.0, preserve_topology=True)

    # Precision reduction
    if _HAVE_SET_PRECISION:
        # Shapely 2.x: grid size ~1e-6 degrees (~0.11 m at equator)
        g["geometry"] = g.geometry.apply(lambda geom: _set_precision(geom, grid_size=1e-6))
    else:
        # Shapely 1.x: round coordinates via transform
        import numpy as _np
        def _rounder(x, y, z=None, dec=6):
            rx = _np.round(x, dec); ry = _np.round(y, dec)
            if z is None:
                return rx, ry
            return rx, ry, _np.round(z, dec)
        try:
            g["geometry"] = g.geometry.apply(lambda geom: _shp_transform(_rounder, geom))
        except Exception:
            pass  # keep simplified geometry as-is if transform fails
    return g

# ----------------- build geometry once -----------------

def build_geometry_geojson() -> dict:
    layer = _find_sa2_layer(ASGS_SA2_GPKG)
    gdf = gpd.read_file(ASGS_SA2_GPKG, layer=layer)
    cols = list(gdf.columns)
    code_col = _first_present(cols, ["SA2_MAINCODE_2021", "SA2_CODE_2021", "SA2_CODE21"])
    name_col = next((c for c in ["SA2_NAME_2021", "SA2_NAME21"] if c in cols), None)

    gdf = gdf[[code_col, *( [name_col] if name_col else [] ), "geometry"]].copy()
    gdf.rename(columns={code_col: "sa2_code"}, inplace=True)
    if name_col:
        gdf.rename(columns={name_col: "sa2_name"}, inplace=True)

    # Keep only WA SA2s (codes start with '5'); ensure str type
    gdf["sa2_code"] = gdf["sa2_code"].astype(str)
    gdf = gdf[gdf["sa2_code"].str.startswith("5")].copy()

    # Simplify a bit for web perf
    gdf = _simplify_in_meters(gdf, tol_m=200)  # increase to 300–500 for smaller files

    # Ensure non-geom fields are strings for GeoJSON
    for c in gdf.columns:
        if c != "geometry":
            gdf[c] = gdf[c].astype(str)

    geojson = json.loads(gdf.to_json())
    _write_json(SITE_DIR / "sa2_wa_simplified.geojson", geojson)
    return geojson

# ----------------- build month data & summary -----------------

def build_monthly_data():
    # Forecast history (append-only parquet maintained by model_forecast.py)
    hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
    if not hist_path.exists():
        # fallback to latest-only
        hist_path = STAGE_DIR / "price_pressure_forecast_sa2.parquet"
    preds = pd.read_parquet(hist_path).copy()
    preds["month"] = _to_month(preds["month"])
    preds["sa2_code"] = preds["sa2_code"].astype(str)

    # Augment with any months present in docs/data/ to avoid losing slider history
    try:
        docs_preds = _read_docs_predictions()
        if not docs_preds.empty:
            docs_preds["month"] = _to_month(docs_preds["month"]).astype("datetime64[ns]")
            preds = (pd.concat([preds, docs_preds], ignore_index=True)
                     .drop_duplicates(subset=["sa2_code", "month"], keep="first"))
    except Exception:
        # Non-fatal: continue with parquet-only preds
        pass

    # Realized labels from SA2 panel (month t compares rent_t vs rent_{t-1})
    sa2 = pd.read_parquet(STAGE_DIR / "bonds_panel_sa2.parquet").copy()
    sa2["month"] = _to_month(sa2["month"])
    sa2["sa2_code"] = sa2["sa2_code"].astype(str)
    sa2 = sa2.sort_values(["sa2_code", "month"])
    sa2["rent_prev"] = sa2.groupby("sa2_code")["median_rent"].shift(1)
    sa2["actual"] = (sa2["median_rent"] / sa2["rent_prev"] - 1.0) > RENT_GROWTH_THRESHOLD
    labels = sa2.loc[sa2["rent_prev"].notna(), ["sa2_code", "month", "actual"]]

    # Merge to expose actuals when available
    joined = preds.merge(labels, on=["sa2_code", "month"], how="left")

    # Months for slider = all forecast months (ensures forward-compat)
    months = sorted(set(preds["month"].unique().tolist()))
    month_strs = [pd.Timestamp(m).strftime("%Y-%m") for m in months]
    _write_json(DATA_DIR / "months.json", month_strs)

    # Per-month files: { sa2_code: {p: float|null, y: 0/1/null} }
    for m in months:
        dfm = joined[joined["month"] == m]
        payload = {}
        for _, r in dfm.iterrows():
            p = None if pd.isna(r.get("price_pressure_prob", np.nan)) else float(r["price_pressure_prob"])
            l = None if pd.isna(r.get("prob_p05", np.nan)) else float(r["prob_p05"])
            u = None if pd.isna(r.get("prob_p95", np.nan)) else float(r["prob_p95"])
            y = (None if pd.isna(r.get("actual", np.nan)) else int(bool(r["actual"])))
            payload[str(r["sa2_code"])] = {"p": p, "l": l, "u": u, "y": y}
        _write_json(DATA_DIR / f"{pd.Timestamp(m).strftime('%Y-%m')}.json", payload)

    # Summary metrics (AUC, Brier) per month where actuals exist
    def _roc_auc(y, p):
        y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
        if y.sum() == 0 or (1 - y).sum() == 0:
            return float("nan")
        order = np.argsort(p)
        ranks = np.empty_like(p, dtype=float); ranks[order] = np.arange(1, len(p)+1)
        uniq, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
        if np.any(cnt > 1):
            means = np.bincount(inv, ranks) / cnt[inv]
            ranks = means
        n1 = float(y.sum()); n0 = float(len(y) - y.sum())
        return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def _brier(y, p):
        y = np.asarray(y).astype(float); p = np.asarray(p).astype(float)
        return float(np.mean((p - y) ** 2))
    def _log_loss(y, p, eps: float = 1e-12):
        y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    summary = {}
    for m, dfm in joined.dropna(subset=["actual"]).groupby("month"):
        p = dfm["price_pressure_prob"].to_numpy()
        y = dfm["actual"].astype(int).to_numpy()
        thr = 0.5
        yhat = (p >= thr).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        summary[pd.Timestamp(m).strftime("%Y-%m")] = {
            "n_sa2": int(len(dfm)),
            "base_rate": float(y.mean()),
            "auc": _roc_auc(y, p),
            "brier": _brier(y, p),
            "log_loss": _log_loss(y, p),
            "precision_at_0_5": (tp / (tp + fp)) if (tp + fp) else float("nan"),
            "recall_at_0_5": (tp / (tp + fn)) if (tp + fn) else float("nan"),
            "accuracy_at_0_5": ((tp + tn) / len(dfm)) if len(dfm) else float("nan"),
        }
    _write_json(DATA_DIR / "summary.json", summary)

# ----------------- write index.html -----------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>WA Rental Forecast — SA2 Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<style>
  html, body, #map { height: 100%; margin: 0; }
  .panel {
    position: absolute; top: 10px; left: 10px; z-index: 1000;
    background: rgba(255,255,255,0.94); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15); font-family: system-ui, sans-serif; max-width: 420px;
  }
  .panel h1 { font-size: 16px; margin: 0 0 8px; }
  .row { display:flex; gap: 8px; align-items: center; margin: 6px 0; flex-wrap: wrap;}
  label { font-size: 12px; color:#444; }
  select, input[type=range], button { font-size: 12px; }
  .legend { margin-top: 6px; font-size: 12px; }
  .legend .grad { height: 10px; background: linear-gradient(to right,#ffffcc,#ffeda0,#feb24c,#fd8d3c,#f03b20,#bd0026); }
  .legend .grad-div { height: 10px; background: linear-gradient(to right,#2166ac,#67a9cf,#d1e5f0,#f7f7f7,#fddbc7,#ef8a62,#b2182b); }
  .footer {
    position: absolute; bottom: 8px; left: 10px; z-index: 1000;
    background: rgba(255,255,255,0.9); padding: 6px 8px; border-radius: 6px; font-size: 12px;
  }
  .pill { display:inline-block; padding:2px 6px; border-radius:10px; background:#f1f1f1; margin-left:6px; }
  .tooltip-line { margin:0; }
  .key-btn { margin-left: 6px; }
  .key-panel {
    position: absolute; right: 10px; top: 10px; z-index: 1100;
    background: rgba(255,255,255,0.97); padding: 12px 14px; border-radius: 8px; width: 360px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2); display: none; font-size: 12px;
  }
  .key-panel h2 { margin: 0 0 8px; font-size: 14px; }
  .key-section { margin-bottom: 10px; }
  .swatch { display:inline-block; width:18px; height:12px; border:1px solid #999; margin-right:6px; vertical-align:middle; }
  .gradbar { height: 10px; border:1px solid #999; margin: 4px 0; }
  .key-close { position:absolute; top:6px; right:8px; border:none; background:#eee; border-radius:4px; cursor:pointer; }
</style>
</head>
<body>
<div id="map"></div>

<div class="panel">
  <h1>WA Rental Forecast — SA2</h1>
  <div class="row">
    <label for="metric">Metric</label>
    <select id="metric">
      <option value="forecast">Forecast probability (Pr[Δrent&gt;2%])</option>
      <option value="actual">Actual outcome (rise &gt; 2%)</option>
      <option value="correct60">Correct @ 0.60 (green = correct)</option>
      <option value="error">Error (prob − actual)</option>
      <option value="abserr">Abs. error |prob − actual|</option>
    </select>
  </div>
  <div class="row">
    <label for="monthSel">Month</label>
    <input id="monthRange" type="range" min="0" max="0" step="1" value="0"/>
    <span id="monthLabel" class="pill">—</span>
    <button id="playBtn">▶</button>
  </div>
  <div class="row"></div>
  <div class="legend" id="legend"></div>
</div>

<!-- removed key panel; legend explains active metric -->

<div class="footer">
  <span id="metrics">No evaluation yet for this month.</span>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script>
let map, layer, months=[], monthIdx=0;
let monthData = null;     // { sa2_code: {p: float|null, y: 0|1|null} }
let summary = {};         // { "YYYY-MM": {auc,brier,base_rate,n_sa2} }
const threshold = 0.5;    // default for footer metrics (kept)
const threshold60 = 0.6;  // cutoff for "Correct @ 0.60" metric

function fmtPct(x) {
  const v = Number.parseFloat(x);
  if (!Number.isFinite(v)) return "—";
  return (Math.round(v * 1000) / 10).toFixed(1) + "%";
}
function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }

function colorProb(p){ if(p==null || isNaN(p)) return "#d9d9d9";
  const stops=[[0.0,"#ffffcc"],[0.2,"#ffeda0"],[0.4,"#feb24c"],[0.6,"#fd8d3c"],[0.8,"#f03b20"],[1.0,"#bd0026"]];
  for(let i=1;i<stops.length;i++){ if(p<=stops[i][0]){
    const t=(p-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
    return lerpColor(stops[i-1][1],stops[i][1],t); } }
  return stops[stops.length-1][1];
}
function colorActual(y){ if(y==null) return "#d9d9d9"; return y? "#bd0026" : "#c7e9c0"; }
function colorError(e){ if(e==null || isNaN(e)) return "#d9d9d9";
  const v = clamp((e+1)/2,0,1);
  const stops=[[0,"#2166ac"],[0.25,"#67a9cf"],[0.5,"#f7f7f7"],[0.75,"#ef8a62"],[1,"#b2182b"]];
  for(let i=1;i<stops.length;i++){ if(v<=stops[i][0]){
    const t=(v-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
    return lerpColor(stops[i-1][1],stops[i][1],t); } }
  return stops[stops.length-1][1];
}
function colorAbsErr(ae){ if(ae==null || isNaN(ae)) return "#d9d9d9";
  const v=clamp(ae,0,1);
  const stops=[[0,"#f7fcb9"],[0.33,"#addd8e"],[0.66,"#31a354"],[1,"#006837"]];
  for(let i=1;i<stops.length;i++){ if(v<=stops[i][0]){
    const t=(v-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
    return lerpColor(stops[i-1][1],stops[i][1],t); } }
  return stops[stops.length-1][1];
}
function colorCorrect(correct){
  if(correct==null) return "#d9d9d9";
  return correct ? "#31a354" : "#ef3b2c"; // green for correct, red for incorrect
}
function colorConfusion(kind){
  // kind: 'TP','TN','FP','FN', null -> grey
  switch(kind){
    case 'TP': return '#1a9850'; // dark green
    case 'TN': return '#74add1'; // blue
    case 'FP': return '#fdae61'; // orange
    case 'FN': return '#d73027'; // red
    default: return '#d9d9d9';
  }
}
function lerpColor(a,b,t){
  const pa=parseInt(a.slice(1),16), pb=parseInt(b.slice(1),16);
  const ar=(pa>>16)&0xff, ag=(pa>>8)&0xff, ab=pa&0xff;
  const br=(pb>>16)&0xff, bg=(pb>>8)&0xff, bb=pb&0xff;
  const r=Math.round(ar+(br-ar)*t), g=Math.round(ag+(bg-ag)*t), bl=Math.round(ab+(bb-ab)*t);
  return "#"+((1<<24)+(r<<16)+(g<<8)+bl).toString(16).slice(1);
}

function setLegend(kind){
  const el=document.getElementById("legend");
  if(kind==="forecast"){
    el.innerHTML = `<div><b>Forecast probability</b> (Pr[Δrent &gt; 2%])</div>
      <div class="grad" style="height:10px;background:linear-gradient(to right,#ffffcc,#ffeda0,#feb24c,#fd8d3c,#f03b20,#bd0026)"></div>
      <div style="display:flex;justify-content:space-between;"><span>0</span><span>1</span></div>`;
  } else if(kind==="actual"){
    el.innerHTML = `<div><b>Actual outcome</b></div>
      <div style="display:flex;gap:6px;align-items:center;">
        <span style="display:inline-block;width:18px;height:12px;background:#bd0026;border:1px solid #999"></span> rise &gt; 2%
        <span style="display:inline-block;width:18px;height:12px;background:#c7e9c0;border:1px solid #999;margin-left:10px;"></span> not rise
        <span style="display:inline-block;width:18px;height:12px;background:#d9d9d9;border:1px solid #999;margin-left:10px;"></span> n/a
      </div>`;
  } else if(kind==="correct60"){
    el.innerHTML = `<div><b>Correct @ 0.60</b> (p ≥ 0.60 vs actual)</div>
      <div style="display:flex;gap:6px;align-items:center;">
        <span style="display:inline-block;width:18px;height:12px;background:#31a354;border:1px solid #999"></span> correct
        <span style="display:inline-block;width:18px;height:12px;background:#ef3b2c;border:1px solid #999;margin-left:10px;"></span> incorrect
        <span style="display:inline-block;width:18px;height:12px;background:#d9d9d9;border:1px solid #999;margin-left:10px;"></span> n/a
      </div>`;
  } else if(kind==="error"){
    el.innerHTML = `<div><b>Error</b> (prob − actual)</div>
      <div class="grad-div" style="height:10px;background:linear-gradient(to right,#2166ac,#67a9cf,#f7f7f7,#ef8a62,#b2182b)"></div>
      <div style="display:flex;justify-content:space-between;"><span>-1</span><span>0</span><span>+1</span></div>`;
  } else {
    if(kind==="abserr"){
      el.innerHTML = `<div><b>Absolute error</b> |prob − actual|</div>
        <div class="grad" style="height:10px;background:linear-gradient(to right,#f7fcb9,#addd8e,#31a354,#006837)"></div>
        <div style="display:flex;justify-content:space-between;"><span>0</span><span>1</span></div>`;
    }
  }
}

function styleFeature(feat){
  const code = feat.properties.sa2_code;
  const rec = monthData ? monthData[code] : null;
  const metric = document.getElementById("metric").value;
  let fill="#d9d9d9";
  if(rec){
    if(metric==="forecast") fill = colorProb(rec.p);
    else if(metric==="actual") fill = colorActual(rec.y);
    else if(metric==="correct60"){
      const yhat = (rec.p==null) ? null : (rec.p >= threshold60 ? 1 : 0);
      const ok = (rec.y==null || yhat==null) ? null : (yhat===rec.y);
      fill = colorCorrect(ok);
    }
    else if(metric==="error"){
      const e = (rec.p==null || rec.y==null) ? null : (rec.p - rec.y);
      fill = colorError(e);
    } else if(metric==="abserr"){
      const ae = (rec.p==null || rec.y==null) ? null : Math.abs(rec.p - rec.y);
      fill = colorAbsErr(ae);
    }
  }
  return { fillColor: fill, color: "#666", weight: 0.4, fillOpacity: 0.85 };
}

function tooltipHTML(feat){
  const code = feat.properties.sa2_code;
  const name = feat.properties.sa2_name || "";
  const rec = monthData ? monthData[code] : null;
  const p = rec? rec.p : null, y = rec? rec.y : null;
  const l = rec? rec.l : null, u = rec? rec.u : null;
  const e = (p==null || y==null) ? null : (p - y);
  const ae = (p==null || y==null) ? null : Math.abs(p - y);
  const metric = document.getElementById("metric").value;
  
  const ci = (l==null || u==null) ? '—' : `${fmtPct(l)} – ${fmtPct(u)}`;
  let extra = '';
  if(metric === 'correct60'){
    extra = `
          <p class="tooltip-line">Predicted (≥0.60): ${p==null? "—" : (p>=threshold60? "Raise":"No raise")}</p>
          <p class="tooltip-line">Correct @0.60: ${(y==null || p==null) ? "—" : (((p>=threshold60?1:0)===y) ? "Yes" : "No")}</p>`;
  }
  return `<p class="tooltip-line"><b>${name || code}</b></p>
          <p class="tooltip-line">SA2: ${code}</p>
          <p class="tooltip-line">Forecast: ${fmtPct(p)}</p>
          <p class="tooltip-line">90% CI: ${ci}</p>
          <p class="tooltip-line">Actual rise &gt;2%: ${y==null? "—" : (y? "Yes":"No")}</p>${extra}
          
          <p class="tooltip-line">Error (p−y): ${e==null? "—" : (Math.round(e*1000)/1000)}</p>
          <p class="tooltip-line">|Error|: ${ae==null? "—" : (Math.round(ae*1000)/1000)}</p>`;
}

// Compute tooltip WHEN it opens (so it always uses the latest monthData)
function onEachFeature(feat, l){
  l.bindTooltip('', {sticky:true});
  l.on('tooltipopen', (e) => { e.tooltip.setContent(tooltipHTML(feat)); });
  l.on('mouseover', () => l.setStyle({weight: 2}));
  l.on('mouseout',  () => l.setStyle({weight: 0.4}));
}

async function loadMonth(idx){
  const month = months[idx];
  document.getElementById("monthLabel").textContent = month;
  const res = await fetch(`data/${month}.json?ts=` + Date.now());
  monthData = await res.json().catch(_=> (monthData=null));

  // Footer metrics
  const m = summary[month] || {};
  const el = document.getElementById("metrics");
  const auc   = (typeof m.auc   === 'number' && Number.isFinite(m.auc))   ? m.auc.toFixed(3)   : '—';
  const brier = (typeof m.brier === 'number' && Number.isFinite(m.brier)) ? m.brier.toFixed(3) : '—';
  const base  = (typeof m.base_rate === 'number') ? (m.base_rate*100).toFixed(1)+'%' : '—';
  const n     = (typeof m.n_sa2 === 'number') ? m.n_sa2 : '—';
  const ll   = (typeof m.log_loss === 'number' && Number.isFinite(m.log_loss)) ? m.log_loss.toFixed(3) : '—';
  const prec = (typeof m.precision_at_0_5 === 'number' && Number.isFinite(m.precision_at_0_5)) ? (m.precision_at_0_5*100).toFixed(0)+'%' : '—';
  const rec  = (typeof m.recall_at_0_5 === 'number' && Number.isFinite(m.recall_at_0_5)) ? (m.recall_at_0_5*100).toFixed(0)+'%' : '—';
  const acc  = (typeof m.accuracy_at_0_5 === 'number' && Number.isFinite(m.accuracy_at_0_5)) ? (m.accuracy_at_0_5*100).toFixed(0)+'%' : '—';
  el.textContent = `AUC ${auc} | Brier ${brier} | LogLoss ${ll} | Precision ${prec} | Recall ${rec} | Accuracy ${acc} | Base rate ${base} | n=${n}`;

  // Repaint and refresh any open tooltips
  layer.setStyle(styleFeature);
  layer.eachLayer((ln) => {
    const tt = ln.getTooltip && ln.getTooltip();
    if (tt && ln.isTooltipOpen && ln.isTooltipOpen()) {
      tt.setContent(tooltipHTML(ln.feature));
    }
  });
}

async function init(){
  map = L.map('map').setView([-31.95, 115.86], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{ attribution: '&copy; OpenStreetMap' }).addTo(map);

  // Load months & summary (with cache-busting)
  months  = await (await fetch('data/months.json?ts='  + Date.now())).json();
  summary = await (await fetch('data/summary.json?ts=' + Date.now())).json().catch(_=> ({}));

  // Month slider
  const range = document.getElementById('monthRange');
  range.max = String(Math.max(0, months.length-1));
  range.value = String(Math.max(0, months.length-1));
  document.getElementById('monthLabel').textContent = months.length? months[months.length-1] : '—';
  range.addEventListener('input', e => loadMonth(parseInt(e.target.value,10)));

  // Legend + metric changer
  const metricSel = document.getElementById('metric');
  setLegend(metricSel.value);
  metricSel.addEventListener('change', () => {
    setLegend(metricSel.value);
    layer.setStyle(styleFeature);
    // refresh any open tooltips to reflect new numbers/format
    layer.eachLayer((ln) => {
      const tt = ln.getTooltip && ln.getTooltip();
      if (tt && ln.isTooltipOpen && ln.isTooltipOpen()) {
        tt.setContent(tooltipHTML(ln.feature));
      }
    });
  });

  // Key panel (optional UI) — guard missing DOM nodes
  const keyBtn = document.getElementById('keyBtn');
  const keyPanel = document.getElementById('keyPanel');
  const keyClose = document.getElementById('keyClose');
  if (keyBtn && keyPanel && keyClose) {
    keyBtn.addEventListener('click', () => {
      const t1 = document.getElementById('keyThr1');
      const t2 = document.getElementById('keyThr2');
      if (t1) t1.textContent = '@0.50';
      if (t2) t2.textContent = '@0.50';
      keyPanel.style.display = 'block';
    });
    keyClose.addEventListener('click', () => { keyPanel.style.display = 'none'; });
  }

  // Play/pause
  let playing=false, handle=null;
  const btn = document.getElementById('playBtn');
  btn.addEventListener('click', () => {
    playing = !playing;
    btn.textContent = playing ? "⏸" : "▶";
    if(playing){
      handle = setInterval(() => {
        let i = parseInt(range.value,10);
        i = (i+1) % months.length;
        range.value = String(i);
        loadMonth(i);
      }, 1200);
    } else {
      clearInterval(handle);
    }
  });

  // Geometry layer (cache-busted)
  const geom = await (await fetch('sa2_wa_simplified.geojson?ts=' + Date.now())).json();
  layer = L.geoJSON(geom, { style: styleFeature, onEachFeature }).addTo(map);

  // First paint
  if(months.length) await loadMonth(months.length-1);
}
init();
</script>
</body>
</html>
"""

def write_index():
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "index.html").write_text(INDEX_HTML, encoding="utf-8")

# ----------------- main -----------------

def main():
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    build_geometry_geojson()
    build_monthly_data()
    write_index()
    print(f"Built static site → {SITE_DIR}/ (open docs/index.html)")

if __name__ == "__main__":
    main()
