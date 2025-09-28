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
from src.features.dates import to_month
from src.common.sa2 import load_sa2_names
from src.common.spatial import detect_sa2_fields, find_sa2_layer, simplify_in_meters

# --- Tuning: geometry simplification tolerance (in meters) ---
SIMPLIFY_TOL_M = 150  # ~150 m; raise to 250–500 for a smaller & faster map


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
    preds = pd.read_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet").copy()
    preds["month"] = to_month(preds["month"])
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
    preds["month"] = to_month(preds["month"])
    preds["sa2_code"] = preds["sa2_code"].astype(str)

    # Read just the SA2 layer and filter to WA
    layer = find_sa2_layer(ASGS_SA2_GPKG)
    gdf = gpd.read_file(ASGS_SA2_GPKG, layer=layer)
    # Use shared field detector from src.common.spatial
    code_col, name_col = detect_sa2_fields(gdf)
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
    gdf = simplify_in_meters(gdf, SIMPLIFY_TOL_M)

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

def precision_at_k_latest(k_list: list[int] = [10, 20, 50, 100]) -> pd.DataFrame:
    """Compute precision@K and lift for the latest realized month and write a CSV."""
    # Use evaluation details if available; fall back to latest predictions
    try:
        from src.features.dates import to_month
        import glob
        from pathlib import Path
        # Find latest eval details
        paths = sorted(glob.glob(str(Path('outputs/evaluations') / 'forecast_eval_details_*.csv')))
        if not paths:
            return pd.DataFrame()
        latest_path = paths[-1]
        df = pd.read_csv(latest_path)
        df["month"] = to_month(df["month"])  # ensure Timestamp dtype
        df = df.sort_values("price_pressure_prob", ascending=False).reset_index(drop=True)
        base_rate = float(df["actual_jump"].astype(bool).mean())
        rows = []
        for K in k_list:
            if len(df) >= K:
                pos_k = int(df.iloc[:K]["actual_jump"].astype(bool).sum())
                prec_k = float(pos_k / K)
                lift_k = float(prec_k / base_rate) if base_rate > 0 else float('nan')
                rows.append({"month": df["month"].iloc[0], "k": K, "precision": prec_k, "lift": lift_k, "positives_in_top_k": pos_k, "base_rate": base_rate})
        out = pd.DataFrame(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / 'precision_at_k_latest.csv'
        out.to_csv(path, index=False)
        print(f"Wrote {path}")
        return out
    except Exception:
        return pd.DataFrame()


def write_action_list(k: int = 20) -> pd.DataFrame:
    """Write a top-K action list with SA2 names for the latest month.

    Columns: rank, sa2_code, sa2_name (if available), price_pressure_prob, month
    Output: outputs/tables/action_top{K}.csv
    """
    preds = pd.read_parquet(STAGE_DIR / "price_pressure_forecast_sa2.parquet").copy()
    preds["month"] = to_month(preds["month"])  # normalize type
    latest = preds["month"].max()
    latest_df = preds[preds["month"] == latest].copy()
    latest_df = latest_df.sort_values("price_pressure_prob", ascending=False).head(k).copy()
    names = load_sa2_names(ASGS_SA2_GPKG)
    latest_df["sa2_code"] = latest_df["sa2_code"].astype(str)
    out = latest_df.merge(names, on="sa2_code", how="left")
    # Enrich with latest available house-price snapshot per SA2 (optional)
    try:
        import pandas as pd
        from src.features.dates import to_month
        hp_monthly_path = STAGE_DIR / "house_prices_sa2_monthly.parquet"
        hp_snapshot_path = STAGE_DIR / "house_prices_sa2_snapshot.parquet"
        if hp_monthly_path.exists():
            hp = pd.read_parquet(hp_monthly_path).copy()
        else:
            hp = pd.read_parquet(hp_snapshot_path).copy()
        rename_cols = {}
        if "allocation_weight_sum" in hp.columns:
            rename_cols["allocation_weight_sum"] = "price_allocation_weight_sum"
        if "n_suburbs" in hp.columns:
            rename_cols["n_suburbs"] = "price_suburb_count"
        if rename_cols:
            hp = hp.rename(columns=rename_cols)
        keep_cols = [c for c in ["sa2_code", "month", "median_house_price", "price_allocation_weight_sum", "price_suburb_count"] if c in hp.columns]
        hp = hp[keep_cols].copy()
        hp["sa2_code"] = hp["sa2_code"].astype(str)
        hp["month"] = to_month(hp["month"])  # normalize
        hp_latest = hp.sort_values(["sa2_code", "month"]).groupby("sa2_code", as_index=False).tail(1)
        out = out.merge(hp_latest[[c for c in ["sa2_code", "median_house_price", "price_allocation_weight_sum", "price_suburb_count"] if c in hp_latest.columns]],
                         on="sa2_code", how="left")
    except Exception:
        pass
    out = out.reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    cols_extra: list[str] = []
    for name in ["median_house_price", "price_allocation_weight_sum", "price_suburb_count"]:
        if name in out.columns:
            cols_extra.append(name)
    cols = ["rank", "sa2_code"] + (["sa2_name"] if "sa2_name" in out.columns else []) + ["price_pressure_prob", "month", *cols_extra]
    out = out[cols]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"action_top{k}.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}")
    return out

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
    # Write top-K action lists
    try:
        write_action_list(10)
        write_action_list(20)
    except Exception:
        pass
    barplot_png(top20)

if __name__ == "__main__":
    main()
