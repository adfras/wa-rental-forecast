"""Build SAL→SA2 (2021) mapping for WA from ASGS correspondences.

Input:
  data_raw/asgs2021_correspondences/CG_SAL_2021_SA2_2021.csv

Output:
  data_raw/ssc_to_sa2_2021.csv  (for compatibility, SSC=SAL)
  Columns: ssc_code, ssc_name, sa2_code, sa2_name, allocation_ratio
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    # Accept either SAL or LOCALITY correspondence file name
    candidates = [
        Path("data_raw/asgs2021_correspondences/CG_SAL_2021_SA2_2021.csv"),
        Path("data_raw/asgs2021_correspondences/CG_LOCALITY_2021_SA2_2021.csv"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise SystemExit("Missing SAL/LOCALITY→SA2 correspondence CSV; run abs-correspondences and check contents.")
    df = pd.read_csv(src)
    # Support SAL or LOCALITY column names
    if {"SAL_CODE_2021", "SAL_NAME_2021", "SA2_MAINCODE_2021"} <= set(df.columns):
        col_map = {
            "SAL_CODE_2021": "ssc_code",
            "SAL_NAME_2021": "ssc_name",
            "SA2_MAINCODE_2021": "sa2_code",
            "SA2_NAME_2021": "sa2_name",
            "RATIO_FROM_TO": "allocation_ratio",
        }
    elif {"LOCALITY_PID_2021", "LOCALITY_NAME_2021", "SA2_CODE_2021"} <= set(df.columns):
        col_map = {
            "LOCALITY_PID_2021": "ssc_code",
            "LOCALITY_NAME_2021": "ssc_name",
            "SA2_CODE_2021": "sa2_code",
            "SA2_NAME_2021": "sa2_name",
            "RATIO_FROM_TO": "allocation_ratio",
        }
    else:
        raise SystemExit(f"Unrecognized columns in {src.name}; inspect file to adjust mappings.")

    out = df[list(col_map.keys())].rename(columns=col_map)
    out["sa2_code"] = out["sa2_code"].astype(str)
    out = out[out["sa2_code"].str.startswith("5")].reset_index(drop=True)
    out_path = Path("data_raw/ssc_to_sa2_2021.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out.shape[0]} rows)")


if __name__ == "__main__":
    main()
