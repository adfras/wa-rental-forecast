"""
Extract WA monthly series from ABS Excel downloads you provide.

Usage examples:
  # After manually downloading the ABS spreadsheets:
  python -m tools.abs_extract_from_xlsx \
    --labour data_raw/external/labour-force-timeseries.xlsx \
    --approvals data_raw/external/building-approvals-timeseries.xlsx

This writes (if parsed successfully):
  data_raw/external/unemployment_rate_wa.csv
  data_raw/external/building_approvals_wa.csv

Then run:
  python -m src.data_ingest.external_signals
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data_raw" / "external"


def _parse_unemployment_from_xlsx(path: Path) -> pd.DataFrame | None:
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None
    # Try sheets that look like timeseries first
    sheet_order = sorted(xls.sheet_names, key=lambda s: (0 if "unemp" in s.lower() else 1, s))
    for sheet in sheet_order:
        try:
            dfh = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        # Find a row for Western Australia
        wa_idx = None
        for col in dfh.columns:
            mask = dfh[col].astype(str).str.contains(r"(?i)Western\s+Australia")
            if mask.any():
                wa_idx = mask.idxmax()
                break
        if wa_idx is None:
            continue
        # Candidate date columns
        date_cols = []
        for c in dfh.columns:
            if pd.api.types.is_datetime64_any_dtype(dfh[c]):
                date_cols.append(c)
            else:
                s = str(c)
                if re.search(r"\b(19|20)\d{2}\b", s) or re.search(r"(?i)\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", s):
                    date_cols.append(c)
        if not date_cols:
            continue
        row = dfh.loc[wa_idx, date_cols].T.reset_index()
        row.columns = ["month", "wa_unemp_rate"]
        row["month"] = pd.to_datetime(row["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        row["wa_unemp_rate"] = pd.to_numeric(row["wa_unemp_rate"], errors="coerce")
        row = row.dropna(subset=["month", "wa_unemp_rate"]).sort_values("month")
        if not row.empty:
            return row[["month", "wa_unemp_rate"]]
    return None


def _parse_building_approvals_from_xlsx(path: Path) -> pd.DataFrame | None:
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        # month column guess
        month_col = None
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]) or str(c).lower() in ("date", "month"):
                month_col = c; break
        if month_col is None:
            continue
        # WA column guess
        wa_cols = [c for c in df.columns if re.search(r"(?i)Western\s+Australia", str(c))]
        for wc in wa_cols:
            ser = pd.to_numeric(df[wc], errors="coerce")
            if ser.notna().sum() >= 6:
                out = pd.DataFrame({
                    "month": pd.to_datetime(df[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp(),
                    "wa_building_approvals": ser,
                }).dropna(subset=["month", "wa_building_approvals"]).sort_values("month")
                if not out.empty:
                    return out
    return None


def main(labour_path: str | None, approvals_path: str | None) -> None:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    if labour_path:
        df = _parse_unemployment_from_xlsx(Path(labour_path))
        if df is not None and not df.empty:
            df.to_csv(EXT_DIR / "unemployment_rate_wa.csv", index=False)
            print(f"Wrote {EXT_DIR / 'unemployment_rate_wa.csv'} ({len(df)} rows)")
        else:
            print("Could not extract unemployment rate from provided labour xlsx.")
    if approvals_path:
        df = _parse_building_approvals_from_xlsx(Path(approvals_path))
        if df is not None and not df.empty:
            df.to_csv(EXT_DIR / "building_approvals_wa.csv", index=False)
            print(f"Wrote {EXT_DIR / 'building_approvals_wa.csv'} ({len(df)} rows)")
        else:
            print("Could not extract building approvals from provided approvals xlsx.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract WA monthly series from ABS Excel downloads")
    ap.add_argument("--labour", type=str, default=None, help="Path to labour force XLSX")
    ap.add_argument("--approvals", type=str, default=None, help="Path to building approvals XLSX")
    args = ap.parse_args()
    main(args.labour, args.approvals)
