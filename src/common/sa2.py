"""Helpers for SA2 metadata and lookup tables."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.common.spatial import find_sa2_layer, detect_sa2_fields


def load_sa2_names(path: Path) -> pd.DataFrame:
    """Return a DataFrame with SA2 codes and optional names."""
    layer = find_sa2_layer(path)
    gdf = gpd.read_file(path, layer=layer)
    code_col, name_col = detect_sa2_fields(gdf)

    keep = [code_col]
    if name_col:
        keep.append(name_col)
    out = gdf[keep].copy()
    out.rename(columns={code_col: "sa2_code"}, inplace=True)
    out["sa2_code"] = out["sa2_code"].astype(str)
    if name_col:
        out.rename(columns={name_col: "sa2_name"}, inplace=True)
        out["sa2_name"] = out["sa2_name"].astype(str)
    return out


__all__ = ["load_sa2_names"]
