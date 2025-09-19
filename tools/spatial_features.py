"""
Spatial utilities for SA2 features.

Build a simple SA2 adjacency list for WA from the ASGS SA2 GeoPackage, and
optionally compute neighbor-based features.

Outputs:
  - data_stage/sa2_neighbors.parquet with columns: sa2_code, neighbor

Notes:
  - We consider two SA2s neighbors if their geometries touch (queen contiguity).
  - Only WA SA2s are included (codes starting with '5').
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import geopandas as gpd

from src.common.spatial import detect_sa2_fields, find_sa2_layer
from src.config import STAGE_DIR, ASGS_SA2_GPKG


def build_sa2_adjacency(out_path: Path | None = None) -> pd.DataFrame:
    """Compute SA2 adjacency (queen) for WA and write to parquet.

    Returns a DataFrame with columns: sa2_code, neighbor (both str).
    """
    layer = find_sa2_layer(ASGS_SA2_GPKG)
    gdf = gpd.read_file(ASGS_SA2_GPKG, layer=layer)
    code_col, _ = detect_sa2_fields(gdf)
    gdf = gdf[[code_col, "geometry"]].copy()
    gdf.rename(columns={code_col: "sa2_code"}, inplace=True)
    gdf["sa2_code"] = gdf["sa2_code"].astype(str)
    gdf = gdf[gdf["sa2_code"].str.startswith("5")].reset_index(drop=True)

    # spatial index for speed; drop rows with missing geometry
    gdf = gdf.set_geometry("geometry").copy()
    gdf = gdf[~gdf.geometry.isna()].reset_index(drop=True)
    try:
        sindex = gdf.sindex
    except Exception:
        sindex = None
    pairs = []
    for i, geom in enumerate(gdf.geometry):
        if geom is None or getattr(geom, "is_empty", True):
            continue
        # bounding box candidates (or all if no sindex)
        if sindex is not None:
            cand_idx = list(sindex.intersection(geom.bounds))
        else:
            cand_idx = list(range(len(gdf)))
        for j in cand_idx:
            if j <= i:
                continue
            other = gdf.geometry.iloc[j]
            if other is None or getattr(other, "is_empty", True):
                continue
            if not geom.is_empty and not other.is_empty:
                if geom.touches(other) or geom.intersects(other):
                    pairs.append((gdf.sa2_code.iloc[i], gdf.sa2_code.iloc[j]))
    # undirected -> two rows per pair
    rows = []
    for a, b in pairs:
        if a == b:
            continue
        rows.append({"sa2_code": a, "neighbor": b})
        rows.append({"sa2_code": b, "neighbor": a})
    adj = pd.DataFrame(rows)
    adj = adj.drop_duplicates().reset_index(drop=True)
    if out_path is None:
        out_path = STAGE_DIR / "sa2_neighbors.parquet"
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    adj.to_parquet(out_path, index=False)
    print(f"Wrote SA2 adjacency → {out_path} (rows={len(adj)})")
    return adj


if __name__ == "__main__":
    build_sa2_adjacency()
