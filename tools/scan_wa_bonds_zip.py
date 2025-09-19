"""
Scan a raw WA bonds ZIP and produce:
 - A Markdown report summarizing detected files, columns and inferred months
 - A JSON dictionary of observed normalized column names by table type

This is a diagnostic helper to understand source schema variance before
ingestion. It does not change any downstream behavior.
"""
import re, io, json, zipfile, sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from dateutil import parser as dparser

from typing import Optional

# Try to import project config if available; else fall back
try:
    from src.config import RAW_DIR, STAGE_DIR, REPORT_DIR
except Exception:
    BASE = Path(".").resolve()
    RAW_DIR = BASE / "data_raw"
    STAGE_DIR = BASE / "data_stage"
    REPORT_DIR = BASE / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_DIR.mkdir(parents=True, exist_ok=True)

def norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"__+", "_", c).strip("_")
    return c

def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc, errors="strict")),
                               engine="python", sep=None, on_bad_lines="skip", nrows=500000)
        except Exception:
            continue
    # last resort with replacement
    return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")),
                       engine="python", sep=None, on_bad_lines="skip", nrows=500000)

def parse_month_from_filename(name: str) -> Optional[pd.Timestamp]:
    # 1) full date tokens (prefer last)
    tokens = re.findall(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})", name)
    for tok in reversed(tokens or []):
        try:
            dt = dparser.parse(tok, dayfirst=True, fuzzy=True)
            return pd.Timestamp(dt.year, dt.month, 1)
        except Exception:
            pass
    # 2) YYYY[-_/]MM
    m = re.search(r"(\d{4})[-_/](\d{1,2})\b", name)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return pd.Timestamp(y, mo, 1)
    # 3) YYYY[-_/](Mon)
    m = re.search(r"(\d{4})[-_/](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", name, re.I)
    if m:
        y, mon = int(m.group(1)), m.group(2).title()
        mo = pd.to_datetime(mon, format="%b").month
        return pd.Timestamp(y, mo, 1)
    # 4) (Mon)[-_/]YYYY
    m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_/](\d{4})\b", name, re.I)
    if m:
        mon, y = m.group(1).title(), int(m.group(2))
        mo = pd.to_datetime(mon, format="%b").month
        return pd.Timestamp(y, mo, 1)
    # 5) fallback: whole string
    try:
        dt = dparser.parse(name, dayfirst=True, fuzzy=True)
        return pd.Timestamp(dt.year, dt.month, 1)
    except Exception:
        return None

def classify(name: str) -> str | None:
    s = name.lower()
    if re.search(r"lodg", s):
        return "lodgements"
    if re.search(r"dispos|refund", s):
        return "disposals"
    if re.search(r"post.?code|stock|bonds_by_postcode", s):
        return "stock"
    return None

def main() -> None:
    zips = sorted(RAW_DIR.glob("wa_bonds_*.zip"))
    if not zips:
        print("No bonds ZIP found in data_raw/. Run fetch first.", file=sys.stderr)
        sys.exit(1)
    latest = zips[-1]
    print(f"Scanning {latest.name}")
    schema_observed = {
        "lodgements": defaultdict(int),
        "disposals": defaultdict(int),
        "stock": defaultdict(int)
    }
    file_rows = []
    coverage = []   # per file -> month coverage
    pc_anomalies = defaultdict(int)

    with zipfile.ZipFile(latest, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/") or name.startswith("__MACOSX/") or "/._" in name:
                continue
            kind = classify(name)
            if not kind:
                continue
            raw = zf.read(name)
            try:
                df = read_csv_bytes(raw)
            except Exception as e:
                file_rows.append({"file": name, "kind": kind, "error": str(e)})
                continue

            # normalized + original headers
            orig_cols = list(df.columns)
            normed = df.copy()
            normed.columns = [norm_col(c) for c in normed.columns]

            # store header counts for dictionary
            for c in normed.columns:
                schema_observed[kind][c] += 1

            # try infer month: column or filename
            month_col = None
            for c in ["month", "period", "date", "lodgement_date", "bond_lodgement_date",
                      "disbursed_date", "disposal_date", "refund_date", "end_date"]:
                if c in normed.columns:
                    month_col = c
                    break
            if month_col:
                mon = normed[month_col].dropna().map(lambda v: dparser.parse(str(v), dayfirst=True, fuzzy=True) if pd.notna(v) else None)
                month = pd.Timestamp(mon.iloc[0].year, mon.iloc[0].month, 1) if len(mon) else parse_month_from_filename(name)
            else:
                month = parse_month_from_filename(name)

            # detect postcode column
            pc_col = None
            for c in ["postcode", "post_code", "poa", "poa_code"]:
                if c in normed.columns:
                    pc_col = c
                    break

            # anomaly counts for postcodes
            if pc_col:
                pc_vals = pd.to_numeric(normed[pc_col], errors="coerce").dropna().astype(int)
                for v, cnt in pc_vals.value_counts().head(10).items():
                    # count odd ones we often see
                    if v < 1000 or v > 9999:
                        pc_anomalies[str(v)] += int(cnt)

            file_rows.append({
                "file": name,
                "kind": kind,
                "rows": len(normed),
                "month_inferred": str(month) if pd.notna(month) else "NaT",
                "has_postcode": pc_col is not None,
                "headers_norm": ", ".join(normed.columns[:12]),
                "headers_raw": ", ".join(orig_cols[:12]),
            })
            coverage.append({"kind": kind, "month": month})

    # Build dictionary table
    dict_rows = []
    for kind, cdict in schema_observed.items():
        for col, n in sorted(cdict.items(), key=lambda kv: (-kv[1], kv[0])):
            dict_rows.append({"table": kind, "observed_column_norm": col, "files_with_column": n})

    dict_df = pd.DataFrame(dict_rows)
    files_df = pd.DataFrame(file_rows)
    cov_df = pd.DataFrame(coverage)
    # Month coverage pivot
    if not cov_df.empty:
        cov_df = cov_df[cov_df["month"].notna()]
        cov_piv = (cov_df.groupby(["kind", "month"]).size()
                          .unstack(0).fillna(0).astype(int).sort_index())
    else:
        cov_piv = pd.DataFrame()

    # Save machine-readable dictionary
    schema_json = STAGE_DIR / "schema_observed.json"
    dict_df.to_json(schema_json, orient="records", indent=2)

    # Write a Markdown report
    md = []
    md.append(f"# AHDAP WA Bonds ZIP scan\n\nLatest file: **{latest.name}**")
    md.append("\n## Summary by file\n")
    if not files_df.empty:
        md.append(files_df.sort_values("file").to_markdown(index=False))
    else:
        md.append("_No files decoded._")

    md.append("\n\n## Observed column dictionary (normalized)\n")
    if not dict_df.empty:
        md.append(dict_df.sort_values(["table","files_with_column"], ascending=[True, False]).to_markdown(index=False))
    else:
        md.append("_No columns observed._")

    md.append("\n\n## Month coverage (files per month)\n")
    if not cov_piv.empty:
        md.append(cov_piv.to_markdown())
    else:
        md.append("_No months inferred._")

    if pc_anomalies:
        md.append("\n\n## Frequent postcode anomalies (numeric values outside 1000–9999)\n")
        pc_df = pd.DataFrame([{"postcode_value": k, "approx_rows": v} for k, v in sorted(pc_anomalies.items())])
        md.append(pc_df.to_markdown(index=False))

    out_md = REPORT_DIR / "ahdap_zip_profile.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote:\n - {out_md}\n - {schema_json}")

if __name__ == "__main__":
    main()
23
