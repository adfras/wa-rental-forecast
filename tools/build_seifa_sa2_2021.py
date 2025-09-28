"""Extract SEIFA 2021 IRSAD deciles by SA2 from the ABS Excel.

Usage:
  python -m tools.build_seifa_sa2_2021 --src data_raw/SEIFA_2021_SA2_Indexes.xlsx

Writes:
  data_raw/seifa_sa2_2021.csv with columns: sa2_code, irsad_decile (WA only)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main(src: str) -> None:
    path = Path(src)
    if not path.exists():
        raise SystemExit(f"Missing SEIFA Excel: {path}")
    # Try first sheet by default
    df = pd.read_excel(path, sheet_name=0)
    # Heuristic column detection
    code_col = next((c for c in df.columns if "SA2" in str(c) and "CODE" in str(c).upper()), None)
    irsad_col = next((c for c in df.columns if ("IRSAD" in str(c).upper() and "DEC" in str(c).upper())), None)
    if not code_col or not irsad_col:
        raise SystemExit("Could not find SA2 code and IRSAD decile columns — inspect the Excel and pass correct sheet.")
    out = df[[code_col, irsad_col]].rename(columns={code_col: "sa2_code", irsad_col: "irsad_decile"})
    out["sa2_code"] = out["sa2_code"].astype(str)
    out = out[out["sa2_code"].str.startswith("5")].reset_index(drop=True)
    out_path = Path("data_raw/seifa_sa2_2021.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out.shape[0]} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    main(args.src)

