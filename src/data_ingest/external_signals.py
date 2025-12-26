"""
Fetch or load external monthly signals to augment forecast features.

- RBA cash rate target: programmatic fetch from the RBA page (parse tables).
- ABS/other APIs: optional SDMX‑JSON or CSV URLs via env vars for WA
  unemployment, building approvals, and Perth CPI.
- CoreLogic / other rent indexes: optional CSV URL or local CSV fallback.
- CSV fallbacks: user-supplied files under data_raw/external/.

Writes `data_stage/external_signals.parquet` with columns such as:
  month, rba_cash_rate, wa_unemp_rate_sa, wa_build_approvals_num,
  perth_cpi, perth_cpi_sa, perth_rent_index, perth_rent_index_sa (columns
  present depend on available data).

Environment variables (optional):
- RBA_CASHRATE_CSV_URL          → direct CSV for cash rate target
- ABS_UNEMPLOYMENT_SDMX_URL     → SDMX JSON endpoint for WA unemployment
- ABS_BUILDAPP_SDMX_URL         → SDMX JSON endpoint for WA building approvals
- ABS_PERTH_CPI_SDMX_URL        → SDMX JSON endpoint for Perth CPI (All groups)
- UNEMPLOYMENT_CSV_URL          → fallback CSV for WA unemployment
- BUILDING_APPROVALS_CSV_URL    → fallback CSV for WA building approvals
- PERTH_CPI_CSV_URL             → fallback CSV for Perth CPI
- PERTH_RENT_INDEX_CSV_URL      → CSV for Perth/CoreLogic rent index (or local)
- CORELOGIC_RENT_CSV_URL        → alias for rent index CSV (takes precedence)

If neither SDMX nor CSV URLs are provided, the loader falls back to CSVs
in `data_raw/external/` (files may be empty; they will be ignored).
"""
from __future__ import annotations

from pathlib import Path
import os
import warnings
import io
import csv
import numpy as np
import pandas as pd
import json
import subprocess
import requests

from src.config import RAW_DIR, STAGE_DIR
from src.features.dates import to_month


def _load_env_file() -> None:
    """Load simple KEY=VALUE lines from a .env-style file if present.

    Supported paths (first found is used):
      - .env (repo root)
      - scripts/external_signals.env
    Lines starting with '#' are ignored; shell-style quotes are stripped.
    """
    for rel in (Path.cwd() / ".env", Path.cwd() / "scripts" / "external_signals.env"):
        try:
            if not rel.exists():
                continue
            for line in rel.read_text().splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '=' not in s:
                    continue
                k, v = s.split('=', 1)
                k = k.strip()
                v = v.strip().strip('\"').strip("\'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
            break
        except Exception:
            continue

EXT_DIR = RAW_DIR / "external"
ABS_DATA_BASE = "https://data.api.abs.gov.au/data"
ABS_SDMX_ACCEPT = "application/vnd.sdmx.data+json;version=1.0"



def fetch_rba_cash_rate() -> pd.DataFrame:
    """Scrape the RBA cash-rate page and return monthly target cash rate.

    Returns columns: month (Timestamp), rba_cash_rate (float, percent).
    """
    import pandas as pd  # ensure pandas imported in runtime env

    # Prefer explicit CSV URL if provided via env
    csv_url = os.getenv("RBA_CASHRATE_CSV_URL")
    if csv_url:
        df = _fetch_rba_cashrate_csv(csv_url)
        if not df.empty:
            return df[["month", "rba_cash_rate"]]

    # Fallback: parse the public web page tables
    url = "https://www.rba.gov.au/statistics/cash-rate/"
    try:
        tables = pd.read_html(url)
    except Exception as e:
        warnings.warn(f"Failed to read RBA cash rate page: {e}")
        return pd.DataFrame(columns=["month", "rba_cash_rate"]).astype({"month": "datetime64[ns]"})

    # Heuristic: look for a table containing the target cash rate history
    candidates = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("target" in c for c in cols) or any("cash rate" in c for c in cols):
            candidates.append(t)
    if not candidates:
        # fallback: take the largest table by rows
        candidates = sorted(tables, key=lambda df: len(df), reverse=True)[:1]

    df = candidates[0].copy()
    # Normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Try common shapes: columns like ['date', 'target cash rate (%)'] or similar
    date_col = next((c for c in df.columns if 'date' in c or 'effective' in c), None)
    rate_col = next((c for c in df.columns if 'target' in c or 'cash rate' in c), None)
    if date_col is None or rate_col is None:
        # Best-effort: use first two columns
        cols = list(df.columns)
        if len(cols) >= 2:
            date_col, rate_col = cols[0], cols[1]
        else:
            return pd.DataFrame(columns=["month", "rba_cash_rate"]).astype({"month": "datetime64[ns]"})

    out = df[[date_col, rate_col]].rename(columns={date_col: "date", rate_col: "rate"}).copy()
    # Coerce numerics; the page may include strings like '5.25'
    out["rba_cash_rate"] = pd.to_numeric(out["rate"].astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False), errors="coerce")
    out["month"] = to_month(out["date"], errors="coerce")  # normalize to month
    out = out.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="last")
    return out[["month", "rba_cash_rate"]].sort_values("month").reset_index(drop=True)


def fetch_google_trends() -> pd.DataFrame:
    """Fetch Google Trends interest for rental-related queries in WA and resample monthly.

    Returns columns: month, gt_rent_intent, gt_vacancy_intent (normalized 0-1).
    Silently returns an empty frame if pytrends is unavailable or rate-limited.
    """
    try:
        from pytrends.request import TrendReq
    except Exception:
        return pd.DataFrame(columns=["month", "gt_rent_intent", "gt_vacancy_intent"]).astype({"month": "datetime64[ns]"})

    geo = os.getenv("GT_GEO", "AU-WA")
    # default from 2018 to today to match current model horizon
    timeframe = os.getenv("GT_TIMEFRAME", f"2018-01-01 {pd.Timestamp.today().strftime('%Y-%m-%d')}")
    rent_kw = [k.strip() for k in os.getenv("GT_RENT_KW", "rent,rental,house for rent").split(',') if k.strip()]
    vac_kw  = [k.strip() for k in os.getenv("GT_VACANCY_KW", "rental vacancy,vacancy rate").split(',') if k.strip()]

    def _pull(keywords: list[str]) -> pd.DataFrame:
        if not keywords:
            return pd.DataFrame(columns=["month", "value"]).astype({"month": "datetime64[ns]"})
        try:
            tr = TrendReq(hl="en-US", tz=600)
            tr.build_payload(keywords, geo=geo, timeframe=timeframe)
            df = tr.interest_over_time()
            if df is None or df.empty:
                return pd.DataFrame(columns=["month", "value"]).astype({"month": "datetime64[ns]"})
            df = df.drop(columns=[c for c in df.columns if c == "isPartial"], errors="ignore")
            vals = df.sum(axis=1)
            out = vals.to_frame("value").reset_index().rename(columns={"date": "date"})
            out["month"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
            out = out.groupby("month", as_index=False)["value"].mean()
            out["value"] = out["value"].astype(float)
            return out[["month", "value"]]
        except Exception:
            return pd.DataFrame(columns=["month", "value"]).astype({"month": "datetime64[ns]"})

    rent = _pull(rent_kw).rename(columns={"value": "gt_rent_intent"})
    vac  = _pull(vac_kw).rename(columns={"value": "gt_vacancy_intent"})
    if rent.empty and vac.empty:
        return pd.DataFrame(columns=["month", "gt_rent_intent", "gt_vacancy_intent"]).astype({"month": "datetime64[ns]"})
    df = rent.merge(vac, on="month", how="outer").sort_values("month")
    # normalize each column 0-1 for stability across refetches
    for c in ["gt_rent_intent", "gt_vacancy_intent"]:
        if c in df:
            s = df[c].astype(float)
            rng = (s.max() - s.min()) or 1.0
            df[c] = (s - s.min()) / rng
    return df


def _fetch_csv_url(url: str, month_col: str = "month", value_col: str = "value",
                   rename_to: str | None = None) -> pd.DataFrame:
    """Fetch a CSV URL into a {month, value}-like DataFrame.
    - month_col: name of the date column in the CSV (will be normalized to month)
    - value_col: numeric column to take as the value
    - rename_to: if provided, rename value column to this name
    """
    try:
        with requests.Session() as s:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        warnings.warn(f"CSV fetch failed for {url}: {e}")
        return pd.DataFrame(columns=["month", rename_to or value_col])
    # Normalize
    if month_col not in df.columns:
        # Try to guess a date-like column
        cand = None
        for c in df.columns:
            cl = str(c).lower()
            if "month" in cl or "date" in cl or "period" in cl:
                cand = c; break
        if cand is None:
            warnings.warn(f"CSV at {url} missing a month/date column; cols={list(df.columns)[:6]}")
            return pd.DataFrame(columns=["month", rename_to or value_col])
        month_col = cand
    if value_col not in df.columns:
        # pick the first numeric column
        nc = None
        for c in df.columns:
            if c == month_col: continue
            if pd.api.types.is_numeric_dtype(df[c]):
                nc = c; break
        if nc is None:
            warnings.warn(f"CSV at {url} missing a numeric value column")
            return pd.DataFrame(columns=["month", rename_to or value_col])
        value_col = nc
    out = df[[month_col, value_col]].rename(columns={month_col: "month"}).copy()
    out["month"] = to_month(out["month"], errors="coerce")  # normalize to month
    out = out.dropna(subset=["month"]) 
    if rename_to:
        out = out.rename(columns={value_col: rename_to})
    return out.sort_values("month").reset_index(drop=True)


def _fetch_rba_cashrate_csv(url: str) -> pd.DataFrame:
    """Parse the RBA F1.1 CSV into month + cash-rate columns.

    The published CSV contains metadata rows (Title/Description/etc.) followed by
    data rows with the first column as a date string (DD/MM/YYYY). This helper
    trims the metadata, locates the "Cash Rate Target" column, and returns a
    tidy frame. Missing values (pre-1990, when the target was not published)
    are left as NaN so downstream logic can drop or forward-fill as needed.
    """
    try:
        with requests.Session() as s:
            resp = s.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.text
    except Exception as exc:
        warnings.warn(f"RBA cash-rate CSV fetch failed for {url}: {exc}")
        return pd.DataFrame(columns=["month", "rba_cash_rate"])

    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    idx_cash: int | None = None
    records: list[dict[str, object]] = []

    for row in reader:
        if not row:
            continue
        first = (row[0] or "").strip()
        if not header:
            if first.lower() == "title":
                header = [c.strip() for c in row]
                try:
                    idx_cash = next(i for i, name in enumerate(header) if name.lower().startswith("cash rate target"))
                except StopIteration:
                    idx_cash = None
                continue
            else:
                # Skip metadata before the title row
                continue
        if first.lower() == "series id":
            # Skip the series-id row; data rows follow
            continue
        if not first:
            continue
        if not first[0].isdigit():
            continue
        # Detect YYYY/MM/DD style rows (historical dates)
        # Use pandas later to coerce; here we just collect the string.
        value = None
        if idx_cash is not None and idx_cash < len(row):
            cell = (row[idx_cash] or "").strip()
            if cell:
                value = cell.replace(",", "")
        records.append({"month": first, "rba_cash_rate": value})

    if not records:
        return pd.DataFrame(columns=["month", "rba_cash_rate"])

    df = pd.DataFrame(records)
    df["month"] = pd.to_datetime(df["month"], errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()
    df["rba_cash_rate"] = pd.to_numeric(df["rba_cash_rate"], errors="coerce")
    df = df.dropna(subset=["month"])  # allow NaN rates for pre-1990 months
    return df.sort_values("month").reset_index(drop=True)


def _fetch_abs_series_via_curl(dataflow: str, key: str, *, start_period: str = "2010-01",
                               end_period: str | None = None) -> pd.DataFrame:
    """Fetch an ABS SDMX series via curl to avoid API user-agent blocks."""
    params = [f"startPeriod={start_period}"]
    if end_period:
        params.append(f"endPeriod={end_period}")
    query = "&".join(params)
    url = f"{ABS_DATA_BASE}/{dataflow}/{key}"
    if query:
        url = f"{url}?{query}"
    cmd = [
        "curl",
        "-sS",
        "-H",
        f"Accept: {ABS_SDMX_ACCEPT}",
        url,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    except subprocess.CalledProcessError as exc:
        warnings.warn(f"ABS curl failed for {url}: {exc.stderr.strip() if exc.stderr else exc}")
        return pd.DataFrame(columns=["month", "value"])
    except Exception as exc:  # pragma: no cover - generic safety
        warnings.warn(f"ABS curl error for {url}: {exc}")
        return pd.DataFrame(columns=["month", "value"])

    payload = res.stdout.strip()
    if not payload or payload.startswith("NoRecordsFound"):
        warnings.warn(f"ABS returned no records for {url}")
        return pd.DataFrame(columns=["month", "value"])
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        warnings.warn(f"ABS payload not JSON for {url}: {exc}")
        return pd.DataFrame(columns=["month", "value"])
    return _parse_abs_sdmx_json(obj)


def _parse_abs_sdmx_json(obj: dict) -> pd.DataFrame:
    """Parse a minimal SDMX‑JSON response into a long frame with columns:
    month, value. Assumes a single series is returned (or uses the first).
    """
    try:
        payload = obj.get("data") or obj
        ds_list = payload.get("dataSets") or obj.get("dataSets")
        struct = payload.get("structure") or obj.get("structure")
        if not ds_list or not struct:
            raise KeyError("Missing dataSets/structure")
        ds = ds_list[0]
        series_map = ds.get("series") or {}
        if not series_map:
            # Some providers put observations at top level
            obs = ds.get("observations", {})
            series_map = {"0": {"observations": obs}}
        # observation time values
        obs_dim = struct["dimensions"]["observation"][0]  # TIME_PERIOD
        time_vals = [v.get("id") or v.get("name") for v in obs_dim.get("values", [])]
        # Take first series
        key, s = next(iter(series_map.items()))
        obs = s.get("observations", {})
        rows = []
        for k, arr in obs.items():
            # k is an index as str; arr is [value, ...] or [value]
            try:
                idx = int(k)
            except Exception:
                continue
            if idx < 0 or idx >= len(time_vals):
                continue
            t = time_vals[idx]
            val = None
            if isinstance(arr, (list, tuple)) and arr:
                val = arr[0]
            else:
                val = arr
            rows.append({"month": t, "value": val})
        if not rows:
            return pd.DataFrame(columns=["month", "value"])
        df = pd.DataFrame(rows)
        df["month"] = to_month(df["month"], errors="coerce")  # normalize
        # coerce numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)
    except Exception as e:
        warnings.warn(f"Failed to parse SDMX‑JSON: {e}")
        return pd.DataFrame(columns=["month", "value"]) 


def _fetch_sdmx_json(url: str) -> pd.DataFrame:
    """Fetch SDMX‑JSON URL and return DataFrame with columns [month, value]."""
    try:
        with requests.Session() as s:
            r = s.get(url, timeout=30, headers={"Accept": "application/vnd.sdmx.data+json, application/json"})
            r.raise_for_status()
            obj = r.json()
        return _parse_abs_sdmx_json(obj)
    except Exception as e:
        warnings.warn(f"SDMX fetch failed for {url}: {e}")
        return pd.DataFrame(columns=["month", "value"]) 


def _load_csv_if_exists(path: Path, col_map: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(col_map.values()))
    df = pd.read_csv(path)
    # Expect a 'month' column; allow alternative name via col_map
    if "month" not in df.columns and "month" in col_map:
        # let mapping handle
        pass
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}).copy()
    if "month" not in df.columns:
        # try to find a date-like column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in str(c).lower():
                df = df.rename(columns={c: "month"})
                break
    if "month" not in df.columns:
        warnings.warn(f"{path} missing a 'month' column; skipping")
        return pd.DataFrame(columns=list(col_map.values()))
    df["month"] = to_month(df["month"], errors="coerce")  # normalize month
    # keep only mapped target columns (ensure unique to avoid duplicates)
    keep = ["month"]
    for v in col_map.values():
        if v != "month" and v not in keep:
            keep.append(v)
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    return df[keep].drop_duplicates(subset=["month"]).sort_values("month").reset_index(drop=True)


def main() -> None:
    _load_env_file()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    EXT_DIR.mkdir(parents=True, exist_ok=True)

    pieces = []

    # 1) RBA cash rate (page parse)
    pieces.append(fetch_rba_cash_rate())

    # Optional user CSV for cash rate (fallback or override)
    rba_csv = _load_csv_if_exists(EXT_DIR / "rba_cash_rate.csv",
                                  {"month": "month", "rba_cash_rate": "rba_cash_rate", "value": "rba_cash_rate"})
    if not rba_csv.empty:
        pieces.append(rba_csv)

    # Google Trends rental search intent (optional)
    try:
        gt = fetch_google_trends()
        if not gt.empty:
            pieces.append(gt)
    except Exception:
        pass

    # 2) WA unemployment — prefer API if provided, else ABS curl helper, else CSV fallback
    unemp_df = pd.DataFrame(columns=["month", "wa_unemp_rate_sa"])
    unemp_sdmx = os.getenv("ABS_UNEMPLOYMENT_SDMX_URL")
    if unemp_sdmx:
        tmp = _fetch_sdmx_json(unemp_sdmx)
        if not tmp.empty:
            unemp_df = tmp.rename(columns={"value": "wa_unemp_rate_sa"})[["month", "wa_unemp_rate_sa"]]

    if unemp_df.empty:
        dataflow = os.getenv("ABS_UNEMPLOYMENT_DATAFLOW", "LF")
        key = os.getenv("ABS_UNEMPLOYMENT_KEY", "M13.3.1599.20.5.M")
        start_period = os.getenv("ABS_UNEMPLOYMENT_START", "2010-01")
        tmp = _fetch_abs_series_via_curl(dataflow=dataflow, key=key, start_period=start_period)
        if not tmp.empty:
            unemp_df = tmp.rename(columns={"value": "wa_unemp_rate_sa"})[["month", "wa_unemp_rate_sa"]]

    if unemp_df.empty:
        unemp_csv_url = os.getenv("UNEMPLOYMENT_CSV_URL")
        if unemp_csv_url:
            tmp = _fetch_csv_url(unemp_csv_url, rename_to="wa_unemp_rate")
            if not tmp.empty:
                unemp_df = tmp.rename(columns={"wa_unemp_rate": "wa_unemp_rate_sa"})[["month", "wa_unemp_rate_sa"]]

    if unemp_df.empty:
        tmp = _load_csv_if_exists(EXT_DIR / "unemployment_rate_wa.csv",
                                  {"month": "month", "value": "wa_unemp_rate_sa", "wa_unemp_rate": "wa_unemp_rate_sa"})
        if not tmp.empty:
            unemp_df = tmp
    if not unemp_df.empty:
        pieces.append(unemp_df)

    # 3) WA building approvals — prefer API if provided, else CSV, else try SDMX candidates
    appr_df = pd.DataFrame(columns=["month", "wa_build_approvals_num"])
    appr_sdmx = os.getenv("ABS_BUILDAPP_SDMX_URL")
    if appr_sdmx:
        tmp = _fetch_sdmx_json(appr_sdmx)
        if not tmp.empty:
            appr_df = tmp.rename(columns={"value": "wa_build_approvals_num"})[["month", "wa_build_approvals_num"]]

    if appr_df.empty:
        dataflow = os.getenv("ABS_BUILDAPP_DATAFLOW", "BA_GCCSA")
        key = os.getenv("ABS_BUILDAPP_KEY", "1.1.9.TOT.TOT.10.5GPER.M")
        start_period = os.getenv("ABS_BUILDAPP_START", "2015-01")
        tmp = _fetch_abs_series_via_curl(dataflow=dataflow, key=key, start_period=start_period)
        if not tmp.empty:
            appr_df = tmp.rename(columns={"value": "wa_build_approvals_num"})[["month", "wa_build_approvals_num"]]

    if appr_df.empty:
        appr_csv_url = os.getenv("BUILDING_APPROVALS_CSV_URL")
        if appr_csv_url:
            tmp = _fetch_csv_url(appr_csv_url, rename_to="wa_building_approvals")
            if not tmp.empty:
                appr_df = tmp.rename(columns={"wa_building_approvals": "wa_build_approvals_num"})[["month", "wa_build_approvals_num"]]

    if appr_df.empty:
        tmp = _load_csv_if_exists(EXT_DIR / "building_approvals_wa.csv",
                                  {"month": "month", "value": "wa_build_approvals_num", "wa_building_approvals": "wa_build_approvals_num"})
        if not tmp.empty:
            appr_df = tmp
    if not appr_df.empty:
        pieces.append(appr_df)

    # 4) Perth CPI (headline) — prefer API/CSV, else local fallback
    cpi_df = pd.DataFrame(columns=["month", "perth_cpi"])
    cpi_sdmx = os.getenv("ABS_PERTH_CPI_SDMX_URL")
    if cpi_sdmx:
        tmp = _fetch_sdmx_json(cpi_sdmx)
        if not tmp.empty:
            cpi_df = tmp.rename(columns={"value": "perth_cpi"})[["month", "perth_cpi"]]

    if cpi_df.empty:
        dataflow = os.getenv("ABS_PERTH_CPI_DATAFLOW", "CPI")
        key = os.getenv("ABS_PERTH_CPI_KEY", "1.10001.10.5.M")
        if key:
            start_period = os.getenv("ABS_PERTH_CPI_START", "2010-01")
            tmp = _fetch_abs_series_via_curl(dataflow=dataflow, key=key, start_period=start_period)
            if not tmp.empty:
                cpi_df = tmp.rename(columns={"value": "perth_cpi"})[["month", "perth_cpi"]]

    if cpi_df.empty:
        cpi_csv_url = os.getenv("PERTH_CPI_CSV_URL") or os.getenv("CPI_CSV_URL")
        if cpi_csv_url:
            tmp = _fetch_csv_url(cpi_csv_url, rename_to="perth_cpi")
            if not tmp.empty:
                cpi_df = tmp[["month", "perth_cpi"]]

    if cpi_df.empty:
        tmp = _load_csv_if_exists(EXT_DIR / "perth_cpi.csv",
                                  {"month": "month", "value": "perth_cpi", "perth_cpi": "perth_cpi"})
        if not tmp.empty:
            cpi_df = tmp

    if not cpi_df.empty:
        pieces.append(cpi_df)

    # Perth CPI – rents subcomponent (monthly indicator) to mirror CoreLogic rents trend
    cpi_rent_df = pd.DataFrame(columns=["month", "perth_rent_cpi"])
    cpi_rent_sdmx = os.getenv("ABS_PERTH_RENTS_SDMX_URL")
    if cpi_rent_sdmx:
        tmp = _fetch_sdmx_json(cpi_rent_sdmx)
        if not tmp.empty:
            cpi_rent_df = tmp.rename(columns={"value": "perth_rent_cpi"})[["month", "perth_rent_cpi"]]

    if cpi_rent_df.empty:
        dataflow = os.getenv("ABS_PERTH_RENTS_DATAFLOW", "CPI")
        start_period = os.getenv("ABS_PERTH_RENTS_START", "2010-01")
        rent_key_env = os.getenv("ABS_PERTH_RENTS_KEY")
        rent_keys = [rent_key_env] if rent_key_env else ["1.115522.10.5.M", "1.30014.10.5.M"]
        for rent_key in [k for k in rent_keys if k]:
            tmp = _fetch_abs_series_via_curl(dataflow=dataflow, key=rent_key, start_period=start_period)
            if not tmp.empty:
                cpi_rent_df = tmp.rename(columns={"value": "perth_rent_cpi"})[["month", "perth_rent_cpi"]]
                break

    if cpi_rent_df.empty:
        tmp = _load_csv_if_exists(EXT_DIR / "perth_rent_cpi.csv",
                                  {"month": "month", "value": "perth_rent_cpi", "perth_rent_cpi": "perth_rent_cpi"})
        if not tmp.empty:
            cpi_rent_df = tmp

    if not cpi_rent_df.empty:
        pieces.append(cpi_rent_df)

    # 5) Perth/CoreLogic rent index — CSV only (CoreLogic API requires auth)
    rent_df = pd.DataFrame(columns=["month", "perth_rent_index"])
    rent_csv_url = os.getenv("CORELOGIC_RENT_CSV_URL") or os.getenv("PERTH_RENT_INDEX_CSV_URL")
    if rent_csv_url:
        tmp = _fetch_csv_url(rent_csv_url, rename_to="perth_rent_index")
        if not tmp.empty:
            rent_df = tmp[["month", "perth_rent_index"]]

    if rent_df.empty:
        tmp = _load_csv_if_exists(EXT_DIR / "perth_rent_index.csv",
                                  {"month": "month", "value": "perth_rent_index",
                                   "perth_rent_index": "perth_rent_index"})
        if not tmp.empty:
            rent_df = tmp

    if not rent_df.empty:
        pieces.append(rent_df)

    # 6) Bonds-derived WA rent level (weighted average from bonds_panel_sa2)
    bonds_path = STAGE_DIR / "bonds_panel_sa2.parquet"
    if bonds_path.exists():
        try:
            bonds = pd.read_parquet(bonds_path).copy()
            bonds["month"] = to_month(bonds["month"])
            for col in ["median_rent", "stock_bonds"]:
                if col not in bonds.columns:
                    raise ValueError(f"Required column '{col}' missing in {bonds_path}")
            agg = (
                bonds.groupby("month")
                .apply(lambda g: pd.Series({
                    "wa_bonds_rent_avg": (g["median_rent"] * g["stock_bonds"].fillna(0.0)).sum()
                                           / max(g["stock_bonds"].fillna(0.0).sum(), 1.0)
                }))
                .reset_index()
            )
            base = agg["wa_bonds_rent_avg"].iloc[0]
            if pd.notna(base) and base > 0:
                agg["wa_bonds_rent_index"] = agg["wa_bonds_rent_avg"] / base * 100.0
            pieces.append(agg)
        except Exception as exc:
            warnings.warn(f"Failed to derive bonds-based rent index: {exc}")

    if not pieces:
        print("No external signals found/fetched. Provide API URLs via env vars or CSVs in data_raw/external/.")
        df = pd.DataFrame(columns=["month", "rba_cash_rate", "wa_unemp_rate_sa", "wa_build_approvals_num"])  # empty schema
    else:
        # Outer-join on month; drop duplicate value columns if a later piece
        # carries the same field name (keep the first encountered).
        df = pieces[0]
        for p in pieces[1:]:
            dup_cols = [c for c in p.columns if c != "month" and c in df.columns]
            if dup_cols:
                p = p.drop(columns=dup_cols)
            df = df.merge(p, on="month", how="outer")
        df = df.sort_values("month").reset_index(drop=True)

    out = STAGE_DIR / "external_signals.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(df)} rows and columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
