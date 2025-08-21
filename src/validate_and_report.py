# src/validate_and_report.py
# Top-20 CSV, fast Folium SA2 map, and static bar chart PNG (optimized for speed).
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import branca.colormap as cm

from src.config import (
    STAGE_DIR,
    OUT_DIR,
    FIG_DIR,
    REPORT_DIR,
    ASGS_SA2_GPKG,
    RENT_GROWTH_THRESHOLD,
)

# --- Tuning: geometry simplification tolerance (in meters) ---
SIMPLIFY_TOL_M = 150  # ~150 m; raise to 250–500 for a smaller & faster map

# --- Helpers -------------------------------------------------

def _detect_sa2_fields(gdf: gpd.GeoDataFrame) -> tuple[str, str | None]:
    """Return (sa2_code_col, sa2_name_col_or_None) using canonical ASGS21 names."""
    cols = set(gdf.columns)
    code_candidates = ["SA2_MAINCODE_2021", "SA2_CODE_2021", "SA2_CODE21"]
    name_candidates = ["SA2_NAME_2021", "SA2_NAME21"]
    code_col = next((c for c in code_candidates if c in cols), None)
    if code_col is None:
        raise ValueError(
            "Could not find an SA2 code column in the GeoPackage. "
            f"Tried {code_candidates}. Found: {list(gdf.columns)[:12]} ..."
        )
    name_col = next((c for c in name_candidates if c in cols), None)
    return code_col, name_col


def _simplify_in_meters(gdf: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    """
    Simplify geometry in a projected CRS (meters) for numeric stability, then return to WGS84.
    """
    g = gdf.copy()
    # Project to Australian Albers (EPSG:3577, meters) for simplification
    try:
        g = g.to_crs(3577)
    except Exception:
        # Fallback: simplify in native CRS (likely degrees). Use small tol.
        tol_m = tol_m / 111_000.0  # approx deg per meter near equator
    g["geometry"] = g.geometry.simplify(tol_m, preserve_topology=True)
    # Back to WGS84 for Leaflet/Folium
    try:
        g = g.to_crs(4326)
    except Exception:
        pass

    # Reduce coordinate precision to shrink JSON (Shapely 2.x fast path)
    try:
        from shapely import wkb
        g["geometry"] = g.geometry.apply(
            lambda geom: wkb.loads(wkb.dumps(geom, rounding_precision=6))
        )
    except Exception:
        # Older Shapely: skip precision rounding
        pass
    return g


def _geojson_minimal(gdf: gpd.GeoDataFrame) -> dict:
    """
    Return a GeoJSON dict with only the provided columns + geometry.
    We also coerce any non-JSON types (e.g., timestamps) to strings.
    """
    clean = gdf.copy()
    # Guard against Timestamp serialization sneaking in
    for c in clean.columns:
        if c != "geometry" and not np.issubdtype(clean[c].dtype, np.number):
            clean[c] = clean[c].astype(str)
    return json.loads(clean.to_json())

# --- Top-20 table --------------------------------------------

def top20_table() -> pd.DataFrame:
    preds = pd.read_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet")
    latest = preds["month"].max()
    latest_df = preds[preds["month"] == latest].copy()
    top20 = latest_df.sort_values("price_pressure_prob", ascending=False).head(20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "top20_pressure_risers.csv"
    top20.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (threshold={RENT_GROWTH_THRESHOLD:.2%}, target_month={latest:%Y-%m})")
    return top20

# --- Fast Folium map -----------------------------------------

def map_forecast():
    # Read predictions (latest month implicitly merged by tooltip)
    preds = pd.read_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet").copy()
    preds["sa2_code"] = preds["sa2_code"].astype(str)

    # Read just the SA2 layer and filter to WA
    gdf = gpd.read_file(ASGS_SA2_GPKG, layer="SA2_2021_AUST_GDA2020")
    code_col, name_col = _detect_sa2_fields(gdf)
    gdf[code_col] = gdf[code_col].astype(str)
    gdf = gdf[gdf[code_col].str.startswith("5")].copy()

    # Merge probability (no month in properties → lighter JSON)
    gdf = gdf.merge(
        preds[["sa2_code", "price_pressure_prob"]],
        left_on=code_col, right_on="sa2_code", how="left",
    ).drop(columns=["sa2_code"])
    gdf["price_pressure_prob"] = gdf["price_pressure_prob"].astype(float)
    gdf["prob_round"] = gdf["price_pressure_prob"].round(3)

    # Keep minimal columns before geometry simplification
    keep_cols = [code_col, "price_pressure_prob", "prob_round", "geometry"]
    if name_col:
        keep_cols.insert(1, name_col)
    gdf = gdf[keep_cols].copy()

    # Simplify polygons (major speed-up)
    gdf = _simplify_in_meters(gdf, SIMPLIFY_TOL_M)

    # Build the map (continuous color scale, single GeoJson layer)
    m = folium.Map(location=[-31.95, 115.86], zoom_start=5, tiles="cartodbpositron")

    # Color scale 0→1
    try:
        colormap = cm.linear.YlOrRd_09.scale(0.0, 1.0)
    except Exception:
        colormap = cm.LinearColormap(
            ["#ffffcc", "#ffeda0", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
            vmin=0.0, vmax=1.0
        )
    colormap.caption = f"Pr(next-month rent > {RENT_GROWTH_THRESHOLD:.0%})"
    colormap.add_to(m)

    # Style function (single layer; no duplicate choropleth)
    def _style(feat):
        p = feat["properties"].get("price_pressure_prob", None)
        fill = colormap(p) if p is not None and not np.isnan(p) else "#d9d9d9"
        return {
            "fillColor": fill,
            "color": "#666666",
            "weight": 0.3,
            "fillOpacity": 0.8 if p is not None and not np.isnan(p) else 0.2,
        }

    # Tooltip fields/aliases
    tooltip_fields = [code_col, "prob_round"]
    tooltip_aliases = ["SA2 code:", "Probability:"]
    if name_col:
        tooltip_fields.insert(0, name_col)
        tooltip_aliases.insert(0, "SA2 name:")

    folium.GeoJson(
        data=_geojson_minimal(gdf),
        name="SA2 price pressure",
        style_function=_style,
        tooltip=folium.features.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            sticky=False,
            localize=True,
            labels=True,
        ),
        highlight_function=lambda feat: {"weight": 2, "color": "#444444"},
        control=False,  # fewer controls → less overhead
    ).add_to(m)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_html = REPORT_DIR / "map_price_pressure.html"
    m.save(out_html)
    print(f"Wrote {out_html}")

# --- Static PNG ----------------------------------------------

def barplot_png(top20_df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "top20_pressure_risers.png"

    fig, ax = plt.subplots(figsize=(9, 7))
    y = top20_df["sa2_code"].astype(str).values
    x = top20_df["price_pressure_prob"].values
    order = np.argsort(-x)
    ax.barh(y[order], x[order])  # default matplotlib colors per your style guide
    ax.invert_yaxis()
    ax.set_xlabel(f"Pr(next-month rent > {RENT_GROWTH_THRESHOLD:.0%})")
    ax.set_ylabel("SA2 code")
    ax.set_title("Top-20 SA2s by price-pressure probability")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_png}")

# --- Main ----------------------------------------------------

def main():
    top20 = top20_table()
    map_forecast()
    barplot_png(top20)

if __name__ == "__main__":
    main()
