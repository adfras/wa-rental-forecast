"""Shared spatial helpers for SA2 geometry handling."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np

try:
    from shapely import set_precision as _set_precision  # type: ignore
    _HAVE_SET_PRECISION = True
except Exception:  # pragma: no cover - Shapely < 2.x fallback
    _HAVE_SET_PRECISION = False
    from shapely.ops import transform as _shp_transform  # type: ignore
    import numpy as _np  # type: ignore


def find_sa2_layer(path: Path, *, preferred: Sequence[str] | None = None) -> str:
    """Return the most likely SA2 layer name within an ASGS GeoPackage."""
    preferred = preferred or ("SA2_2021_AUST_GDA2020", "SA2_2021_AUST_GDA94")
    try:
        from pyogrio import list_layers  # type: ignore

        names = [n[0] if isinstance(n, (list, tuple)) else n for n in list_layers(path)]
        for name in preferred:
            if name in names:
                return name
        for name in names:
            if "SA2" in str(name).upper():
                return name
        if names:
            return names[0]
    except Exception:
        pass
    # Fallback to the first preferred value when discovery fails
    return preferred[0]


def detect_sa2_fields(gdf: gpd.GeoDataFrame) -> tuple[str, str | None]:
    """Identify SA2 code and optional SA2 name columns."""
    cols = set(gdf.columns)
    code_candidates: Iterable[str] = ("SA2_MAINCODE_2021", "SA2_CODE_2021", "SA2_CODE21")
    name_candidates: Iterable[str] = ("SA2_NAME_2021", "SA2_NAME21")

    code_col = next((c for c in code_candidates if c in cols), None)
    if code_col is None:
        raise ValueError(
            "Could not find an SA2 code column; expected one of "
            f"{tuple(code_candidates)}, found {list(gdf.columns)[:8]}"
        )
    name_col = next((c for c in name_candidates if c in cols), None)
    return code_col, name_col


def simplify_in_meters(gdf: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    """Simplify geometries using a metric projection before returning to WGS84."""
    work = gdf.copy()
    try:
        work = work.to_crs(3577)  # Australian Albers
        work["geometry"] = work.geometry.simplify(tol_m, preserve_topology=True)
        work = work.to_crs(4326)
    except Exception:
        # Fallback: simplify directly in degrees (small tolerance conversion)
        work["geometry"] = work.geometry.simplify(tol_m / 111_000.0, preserve_topology=True)

    if _HAVE_SET_PRECISION:
        work["geometry"] = work.geometry.apply(lambda geom: _set_precision(geom, grid_size=1e-6))
    else:  # pragma: no cover - exercised on Shapely 1.x only
        def _rounder(x, y, z=None, dec=6):
            rx = _np.round(x, dec)
            ry = _np.round(y, dec)
            if z is None:
                return rx, ry
            return rx, ry, _np.round(z, dec)

        try:
            work["geometry"] = work.geometry.apply(lambda geom: _shp_transform(_rounder, geom))
        except Exception:
            pass
    return work


__all__ = [
    "detect_sa2_fields",
    "find_sa2_layer",
    "simplify_in_meters",
]
