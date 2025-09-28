"""Fetch SEIFA 2021 (IRSAD deciles) for SA2 via ABS ArcGIS REST (free/open).

Writes data_raw/seifa_sa2_2021.csv with columns: sa2_code, irsad_decile (WA only)
"""
from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import requests

BASE = (
    "https://services-ap1.arcgis.com/ypkPEy1AmwPKGNNv/arcgis/rest/services/"
    "ABS_Socio_Economic_Indexes_for_Areas_SEIFA_by_2021_SA2/FeatureServer/0/query"
)


def _page(offset: int, size: int) -> pd.DataFrame:
    params = {
        "where": "sa2_code_2021 LIKE '5%'",  # WA only
        "outFields": "sa2_code_2021,sa2_name_2021,irsad_aus_decile",
        "returnGeometry": "false",
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": size,
    }
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    feats = j.get("features", [])
    rows = []
    for f in feats:
        attr = f.get("attributes", {})
        rows.append({
            "sa2_code": str(attr.get("sa2_code_2021", "")),
            "sa2_name": attr.get("sa2_name_2021", ""),
            "irsad_decile": attr.get("irsad_aus_decile", None),
        })
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    offset = 0
    size = 1000
    while True:
        df = _page(offset, size)
        if df.empty:
            break
        all_rows.append(df)
        if len(df) < size:
            break
        offset += size
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["sa2_code","irsad_decile"])  # type: ignore
    out = out.dropna(subset=["sa2_code"]).copy()
    out = out[["sa2_code", "irsad_decile"]]
    out_path = Path("data_raw/seifa_sa2_2021.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out.shape[0]} rows)")


if __name__ == "__main__":
    main()

