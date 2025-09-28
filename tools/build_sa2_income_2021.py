"""Extract SA2 median weekly household income from Census 2021.

Two input options (pass one):
  --gpkg data_raw/Census2021_G02_SA2.gpkg  (GeoPackage with G02 topic)
  --excel data_raw/G02_SA2.xlsx            (Excel General Community Profile)

Writes:
  data_raw/sa2_income_2021.csv with columns: sa2_code, median_household_income_weekly (WA only)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def from_gpkg(path: Path) -> pd.DataFrame:
    import geopandas as gpd
    gdf = gpd.read_file(path)
    code_col = next((c for c in gdf.columns if "SA2" in c and "CODE" in c.upper()), None)
    inc_col = next((c for c in gdf.columns if ("MEDIAN" in c.upper() and "HOUSEHOLD" in c.upper() and "INCOME" in c.upper())), None)
    if not code_col or not inc_col:
        raise SystemExit("Could not find SA2 code and income columns in GPKG; pass --excel instead.")
    df = pd.DataFrame(gdf[[code_col, inc_col]]).rename(columns={code_col: "sa2_code", inc_col: "median_household_income_weekly"})
    return df


def from_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    code_col = next((c for c in df.columns if "SA2" in c and "CODE" in c.upper()), None)
    inc_col = next((c for c in df.columns if ("MEDIAN" in c.upper() and "HOUSEHOLD" in c.upper() and "INCOME" in c.upper())), None)
    if not code_col or not inc_col:
        raise SystemExit("Could not find SA2 code and income columns in Excel.")
    df = df[[code_col, inc_col]].rename(columns={code_col: "sa2_code", inc_col: "median_household_income_weekly"})
    return df


def main(gpkg: str | None, excel: str | None) -> None:
    if not gpkg and not excel:
        raise SystemExit("Pass --gpkg or --excel input path.")
    if gpkg:
        df = from_gpkg(Path(gpkg))
    else:
        df = from_excel(Path(excel))
    df["sa2_code"] = df["sa2_code"].astype(str)
    df = df[df["sa2_code"].str.startswith("5")].reset_index(drop=True)
    out = Path("data_raw/sa2_income_2021.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({df.shape[0]} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", default=None)
    ap.add_argument("--excel", default=None)
    args = ap.parse_args()
    main(args.gpkg, args.excel)

