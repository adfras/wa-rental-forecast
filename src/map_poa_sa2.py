# src/map_poa_sa2.py
# Map WA bonds panel from POA (postcode) to SA2 using ABS correspondence (2021).
# Robust to correspondence path being CSV, XLSX, or the ABS mega-ZIP bundle.
from __future__ import annotations

from pathlib import Path
import io
import zipfile
import csv

import numpy as np
import pandas as pd

from src.config import STAGE_DIR, POA_SA2_CORRESP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WA_SA2_PREFIX = "5"  # SA2_CODE_2021 starts with '5' for WA in ASGS 2021
NON_WA_POAS = {"0872", "6798", "6799"}  # Out-of-WA postcodes to drop (admin edges)

# ---------------------------------------------------------------------------
# Small utilities (kept minimal and contextual)
# ---------------------------------------------------------------------------
def _pick(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    """Return the first existing column from `candidates` in `df`."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column; tried {candidates}")
    return None


def _discover_xwalk_and_template(path) -> tuple[pd.DataFrame, dict]:
    """
    Read the ABS POA→SA2 correspondence from whatever `path` points to:
      - Plain CSV
      - Plain Excel (.xlsx/.xls)
      - ABS mega-ZIP containing the correspondences

    Returns:
      (df, tmpl) where tmpl describes the *actual* labels and scaling to use:
        tmpl = {
          'source':         str(path),
          'inner_file':     <name inside zip or None>,
          'poa_col':        <e.g. 'POA_CODE_2021'>,
          'sa2_col':        <e.g. 'SA2_CODE_2021'>,
          'ratio_col':      <e.g. 'RATIO_FROM_TO' or 'RATIO' or 'PERCENTAGE'>,
          'scale_percent':  True if ratio is 0–100 (or contains '%') else False,
          'rows':           int
        }
    """
    p = Path(path)
    inner = None

    def _read_csv_bytes(raw: bytes) -> tuple[pd.DataFrame, dict]:
        # Detect encoding and delimiter from bytes
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                text = raw.decode(enc)
                used_enc = enc
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("latin1", errors="replace")
            used_enc = "latin1"
        try:
            delim = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
        except Exception:
            delim = ","
        df = pd.read_csv(io.StringIO(text), sep=delim, engine="python", dtype=str)
        return df, {"enc": used_enc, "delim": delim}

    def _is_true_xlsx_package(zpath: Path) -> bool:
        """True for a real .xlsx file (which is itself a zip with Excel internals)."""
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                names = set(z.namelist())
                return ("[Content_Types].xml" in names) and any(n.startswith("xl/") for n in names)
        except Exception:
            return False

    # ---- Load df (no renames yet) ----
    if p.suffix.lower() in (".xlsx", ".xls"):
        # It could be an actual Excel workbook OR the ABS mega-ZIP saved as .xlsx.
        if zipfile.is_zipfile(p) and not _is_true_xlsx_package(p):
            # It's the ABS mega-ZIP disguised as .xlsx → open the inner correspondence.
            with zipfile.ZipFile(p, "r") as z:
                names = z.namelist()
                preferred = "CG_POA_2021_SA2_2021.xlsx"
                inner = preferred if preferred in names else next(
                    (n for n in names if ("POA" in n.upper() and "SA2" in n.upper()
                                          and n.lower().endswith((".xlsx", ".xls", ".csv")))),
                    None
                )
                if inner is None:
                    raise SystemExit("No POA↔SA2 file found inside the ABS ZIP.")
                raw = z.read(inner)
                if inner.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(io.BytesIO(raw), dtype=str, engine="openpyxl")
                else:
                    df, _ = _read_csv_bytes(raw)
        else:
            # A normal Excel workbook on disk
            df = pd.read_excel(p, dtype=str, engine="openpyxl")

    elif zipfile.is_zipfile(p):
        # Proper ABS mega-ZIP (.zip)
        with zipfile.ZipFile(p, "r") as z:
            names = z.namelist()
            preferred = "CG_POA_2021_SA2_2021.xlsx"
            inner = preferred if preferred in names else next(
                (n for n in names if ("POA" in n.upper() and "SA2" in n.upper()
                                      and n.lower().endswith((".xlsx", ".xls", ".csv")))),
                None
            )
            if inner is None:
                raise SystemExit("No POA↔SA2 file found inside the ABS ZIP.")
            raw = z.read(inner)
            if inner.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(raw), dtype=str, engine="openpyxl")
            else:
                df, _ = _read_csv_bytes(raw)

    else:
        # Plain CSV on disk
        raw = p.read_bytes()
        df, _ = _read_csv_bytes(raw)

    # ---- Discover actual labels + whether ratio needs scaling ----
    cols_lc = {c.lower(): c for c in df.columns}

    def find(*needles: str) -> str | None:
        for lc, orig in cols_lc.items():
            if all(n in lc for n in needles):
                return orig
        return None

    poa_col = (find("poa", "code") or find("post", "code") or find("poa"))
    sa2_col = (find("sa2", "code") or find("sa2", "main") or find("sa2"))
    # Your file shows RATIO_FROM_TO; also support RATIO and PERCENTAGE
    ratio_col = (find("ratio_from_to") or find("ratio") or find("percentage") or
                 find("percent") or find("pct") or find("alloc") or find("share"))

    if not all([poa_col, sa2_col, ratio_col]):
        raise SystemExit(
            f"[xwalk] Missing required columns. "
            f"Have: {list(df.columns)[:8]}… → poa={poa_col} sa2={sa2_col} ratio={ratio_col}"
        )

    r = pd.to_numeric(df[ratio_col], errors="coerce")
    contains_pct = df[ratio_col].astype(str).str.contains("%", na=False).any()
    # If any value > 1 or it contains '%', treat as 0–100 percentages that need /100.
    try:
        maxv = float(np.nanmax(r.values))
    except Exception:
        maxv = np.nan
    scale_percent = bool(contains_pct or (pd.notna(maxv) and maxv > 1.01))

    tmpl = {
        "source": str(p),
        "inner_file": inner,
        "poa_col": poa_col,
        "sa2_col": sa2_col,
        "ratio_col": ratio_col,
        "scale_percent": scale_percent,
        "rows": int(len(df)),
    }
    return df, tmpl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load inputs ---
    bonds = pd.read_parquet(STAGE_DIR / "bonds_panel_postcode.parquet").copy()

    # Discover the correspondence table and the *actual* column labels
    xwalk_raw, tmpl = _discover_xwalk_and_template(POA_SA2_CORRESP)
    poa_col, sa2_col, ratio_col = tmpl["poa_col"], tmpl["sa2_col"], tmpl["ratio_col"]
    print(
        f"[xwalk] source={tmpl['source']} inner={tmpl['inner_file']} "
        f"poa={poa_col} sa2={sa2_col} ratio={ratio_col} "
        f"scale={'%→ratio' if tmpl['scale_percent'] else 'ratio'} rows={tmpl['rows']}"
    )

    # --- Clean/derive keys & allocation ratio ---
    x = xwalk_raw[[poa_col, sa2_col, ratio_col]].copy()

    # POA: extract up to 4 digits, zfill to 4 (keeps leading zeros like '0011')
    x["poa_code"] = x[poa_col].astype(str).str.extract(r"(\d{1,4})", expand=False).str.zfill(4)

    # SA2: ASGS 2021 SA2_code is 9 digits (filter rows that parse cleanly)
    x["sa2_code"] = x[sa2_col].astype(str).str.extract(r"(\d{9})", expand=False)

    # Numeric ratio; if percentages, scale to [0,1]
    r = pd.to_numeric(x[ratio_col], errors="coerce")
    x["alloc_ratio"] = np.where(tmpl["scale_percent"], r / 100.0, r)

    # Drop rows with missing keys/ratios
    x = x.dropna(subset=["poa_code", "sa2_code", "alloc_ratio"]).copy()

    # Keep WA SA2s and drop out-of-WA POAs
    x = x[x["sa2_code"].str.startswith(WA_SA2_PREFIX)]
    x = x[~x["poa_code"].isin(NON_WA_POAS)]

    # --- Prepare bonds panel keys ---
    b = bonds.copy()
    b["postcode"] = b["postcode"].astype(str).str.extract(r"(\d{1,4})", expand=False).str.zfill(4)
    b["month"] = pd.to_datetime(b["month"]).dt.to_period("M").dt.to_timestamp()

    # Harmonise bonds column names (stick to your existing schema)
    lodge_col = _pick(b, ["count_lodgements", "lodgements_count", "lodge_count", "n_lodgements"], required=False)
    disp_col  = _pick(b, ["count_disposals", "disposals_count", "disp_count", "n_disposals"], required=False)
    stock_col = _pick(b, ["stock_bonds", "bonds_held", "bonds_held_count"], required=True)
    rent_col  = _pick(b, ["median_rent", "rent_median", "weekly_rent_amount_median", "lodge_rent_median"], required=False)
    p90_col   = _pick(b, ["p90_rent", "rent_p90", "weekly_rent_amount_p90"], required=False)
    days_col  = _pick(b, ["mean_days_held", "median_days_bond_held", "days_bond_held_median"], required=False)

    # --- Merge POA→SA2 allocation ---
    m = b.merge(
        x[["poa_code", "sa2_code", "alloc_ratio"]],
        left_on="postcode", right_on="poa_code",
        how="inner"
    ).copy()
    m["alloc_ratio"] = m["alloc_ratio"].fillna(0.0)

    # Ensure numeric for downstream math
    for c in [rent_col, p90_col, days_col, lodge_col, disp_col, stock_col]:
        if c is not None and c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    # --- Build weighted pieces ---
    # Weights for flow-weighted means
    m["lodg_w"] = (m[lodge_col] if lodge_col else 0.0)
    m["disp_w"] = (m[disp_col]  if disp_col  else 0.0)
    m["lodg_w"] = m["lodg_w"].fillna(0.0) * m["alloc_ratio"]
    m["disp_w"] = m["disp_w"].fillna(0.0) * m["alloc_ratio"]

    # Numerators for weighted means
    if rent_col:
        m["rent_num"] = m[rent_col] * m["lodg_w"]
    if p90_col:
        m["rent90_num"] = m[p90_col] * m["lodg_w"]
    if days_col:
        m["days_num"] = m[days_col] * m["disp_w"]

    # Allocated counts and stock (sum of alloc*value)
    m["count_lodgements_alloc"] = m["lodg_w"]
    m["count_disposals_alloc"]  = m["disp_w"]
    m["stock_bonds_alloc"]      = m[stock_col].fillna(0.0) * m["alloc_ratio"]

    # --- Aggregate to SA2×month ---
    keys = ["sa2_code", "month"]
    g = m.groupby(keys, as_index=False)
    out = g.agg(
        count_lodgements=("count_lodgements_alloc", "sum"),
        count_disposals=("count_disposals_alloc", "sum"),
        stock_bonds=("stock_bonds_alloc", "sum"),
        lodg_w=("lodg_w", "sum"),
        disp_w=("disp_w", "sum"),
        rent_num=("rent_num", "sum") if "rent_num" in m.columns else ("alloc_ratio", "sum"),
        rent90_num=("rent90_num", "sum") if "rent90_num" in m.columns else ("alloc_ratio", "sum"),
        days_num=("days_num", "sum") if "days_num" in m.columns else ("alloc_ratio", "sum"),
    )

    # Weighted means (protect against zero weights → NaN)
    if "rent_num" in out.columns:
        out["median_rent"] = out["rent_num"] / out["lodg_w"].replace(0, np.nan)
    if "rent90_num" in out.columns:
        out["p90_rent"] = out["rent90_num"] / out["lodg_w"].replace(0, np.nan)
    if "days_num" in out.columns:
        out["mean_days_held"] = out["days_num"] / out["disp_w"].replace(0, np.nan)

    # Derived metrics at SA2
    out["churn_rate"] = (out["count_lodgements"] + out["count_disposals"]) / out["stock_bonds"].clip(lower=1.0)
    out["net_stock_change"] = out["count_lodgements"] - out["count_disposals"]

    # Sort and compute momentum off SA2-level rent
    out = out.sort_values(keys).reset_index(drop=True)
    if "median_rent" in out.columns:
        out["rent_momentum"] = (
            out.groupby("sa2_code")["median_rent"]
               .pct_change(fill_method=None)
               .fillna(0.0)
        )

    # Save tidy set
    keep = [
        "sa2_code", "month", "median_rent", "p90_rent",
        "count_lodgements", "mean_days_held", "count_disposals",
        "stock_bonds", "churn_rate", "net_stock_change", "rent_momentum"
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    outpath = STAGE_DIR / "bonds_panel_sa2.parquet"
    out.to_parquet(outpath, index=False)
    print(f"Wrote SA2 panel to {outpath} with {len(out):,} rows.")

if __name__ == "__main__":
    main()
