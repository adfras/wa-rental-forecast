"""
Fetch ABS latest-release download links via a headless browser and extract
CSV/XLSX resources, then distill WA monthly series into simple CSV files that
src.external_signals can ingest.

This is a pragmatic workaround for dynamic pages where direct links are not
present in static HTML. It requires Playwright with Chromium.

Outputs (written if parsed successfully):
  - data_raw/external/unemployment_rate_wa.csv
  - data_raw/external/building_approvals_wa.csv

Note: This scraper is best-effort and may need adjustments if ABS page
structure changes.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import re
import sys
import io
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data_raw" / "external"


LABOUR_URL = "https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/latest-release"
BUILDAPP_URL = "https://www.abs.gov.au/statistics/industry/building-and-construction/building-approvals-australia/latest-release"


async def _collect_xlsx_links(url: str) -> list[str]:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        # Some content loads late; give it a moment
        await page.wait_for_timeout(2000)
        links = await page.eval_on_selector_all(
            "a[href$='.xlsx']",
            "els => els.map(e => e.href)",
        )
        await browser.close()
    # Deduplicate and keep absolute URLs
    out = []
    seen = set()
    for href in links:
        if not href or not href.endswith(".xlsx"):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append(href)
    return out


def _download(url: str, out_path: Path) -> None:
    import requests
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _parse_unemployment_from_xlsx(path: Path) -> pd.DataFrame | None:
    """Heuristic extraction: find the first sheet that contains 'Unemployment rate'
    and has a 'Western Australia' row or column, then melt to month,value.
    Returns DataFrame with columns month, wa_unemp_rate.
    """
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=None)
        except Exception:
            continue
        # Look for a cell containing 'Unemployment rate'
        if not (df.astype(str).apply(lambda s: s.str.contains("Unemployment rate", case=False, na=False)).any().any()):
            continue
        # Try to find header row by locating 'Series ID' or a month-like header
        # Heuristic: find first row that has a 'Western Australia' mention
        mask_wa = df.astype(str).apply(lambda s: s.str.fullmatch(r"(?i).*Western\s+Australia.*").fillna(False))
        if not mask_wa.any().any():
            continue
        # Assume this is a time series table where columns to the right are dates
        # Try: set the first non-null row as header if it contains year/month strings
        # Fallback to default parsed header
        df2 = pd.read_excel(path, sheet_name=sheet)
        # Find WA row
        wa_row = None
        for col in df2.columns:
            if df2[col].astype(str).str.contains(r"(?i)Western\s+Australia").any():
                wa_row = df2[df2[col].astype(str).str.contains(r"(?i)Western\s+Australia")].iloc[0]
                break
        if wa_row is None:
            continue
        # Extract columns that look like dates or 'YYYY MMM'
        candidates = []
        for c in df2.columns:
            if re.search(r"\b(19|20)\d{2}\b", str(c)) or re.search(r"(?i)\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", str(c)):
                candidates.append(c)
        if not candidates:
            # maybe the index are dates
            continue
        ts = wa_row[candidates].T.reset_index()
        ts.columns = ["month", "wa_unemp_rate"]
        # Normalize month
        ts["month"] = pd.to_datetime(ts["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        ts["wa_unemp_rate"] = pd.to_numeric(ts["wa_unemp_rate"], errors="coerce")
        ts = ts.dropna(subset=["month", "wa_unemp_rate"]).sort_values("month")
        if not ts.empty:
            return ts[["month", "wa_unemp_rate"]]
    return None


def _parse_building_approvals_from_xlsx(path: Path) -> pd.DataFrame | None:
    """Heuristic extraction for WA building approvals from a time-series workbook.
    Returns month, wa_building_approvals.
    """
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        # Look for a WA series column or row with numeric values across months
        if not any(df.columns.astype(str).str.contains(r"(?i)Western\s+Australia")) and \
           not df.apply(lambda s: s.astype(str).str.contains(r"(?i)Western\s+Australia").any()).any():
            continue
        # Identify month-like column
        month_col = None
        for c in df.columns:
            if (str(c).lower().startswith("month") or str(c).lower() == "date" or
                pd.api.types.is_datetime64_any_dtype(df[c])):
                month_col = c; break
        if month_col is None:
            # Try index or first column if parseable
            month_col = df.columns[0]
        # Identify WA column (units or value) — pick the first numeric column with WA in header
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


async def main() -> None:
    # 1) Collect XLSX links from pages
    lab_links = await _collect_xlsx_links(LABOUR_URL)
    bld_links = await _collect_xlsx_links(BUILDAPP_URL)
    # 2) Download a few likely candidates and parse
    EXT_DIR.mkdir(parents=True, exist_ok=True)

    unemp_out = EXT_DIR / "unemployment_rate_wa.csv"
    appr_out = EXT_DIR / "building_approvals_wa.csv"

    got_unemp = False
    for url in lab_links[:8]:
        tmp = EXT_DIR / ("lab_" + Path(url).name)
        try:
            _download(url, tmp)
            df = _parse_unemployment_from_xlsx(tmp)
            if df is not None and not df.empty:
                df.to_csv(unemp_out, index=False)
                got_unemp = True
                break
        except Exception:
            continue

    got_appr = False
    for url in bld_links[:12]:
        tmp = EXT_DIR / ("bld_" + Path(url).name)
        try:
            _download(url, tmp)
            df = _parse_building_approvals_from_xlsx(tmp)
            if df is not None and not df.empty:
                df.to_csv(appr_out, index=False)
                got_appr = True
                break
        except Exception:
            continue

    print(f"Unemployment CSV: {'OK' if got_unemp else 'MISSING'} → {unemp_out if got_unemp else ''}")
    print(f"Building approvals CSV: {'OK' if got_appr else 'MISSING'} → {appr_out if got_appr else ''}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)

