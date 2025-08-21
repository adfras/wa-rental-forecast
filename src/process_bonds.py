# src/process_bonds.py
import re, io, zipfile, warnings
import numpy as np
import pandas as pd
from dateutil import parser as dparser
from src.config import RAW_DIR, STAGE_DIR
from src.column_templates import SCHEMA_TEMPLATES

# ---------- helpers ----------
def _standardise_cols(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"[^\w]+", "_", c)         # spaces, slashes, punctuation -> underscores
        c = re.sub(r"__+", "_", c).strip("_") # fold multiple _
        return c
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    return df

def _parse_month(val):
    if pd.isna(val):
        return pd.NaT
    try:
        dt = dparser.parse(str(val), dayfirst=True, fuzzy=True)
        return pd.Timestamp(dt.year, dt.month, 1)
    except Exception:
        return pd.NaT

def _parse_month_from_filename(name: str):
    """
    Extract a month from the filename.
    Supports:
      - '01-07-2025-31-07-2025'
      - '2025-07', '2025_07', '2025/07'
      - '2025-Jul', 'Jul-2025', '2024-Sep', etc.
    Returns first-of-month Timestamp or NaT.
    """
    # 1) Exact date tokens (DD-MM-YYYY, YYYY-MM-DD, etc.)
    tokens = re.findall(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})", name)
    for tok in reversed(tokens or []):  # prefer the last date-like token
        try:
            dt = dparser.parse(tok, dayfirst=True, fuzzy=True)
            return pd.Timestamp(dt.year, dt.month, 1)
        except Exception:
            pass

    # 2) Year + numeric month (YYYY[-_/]MM)
    m = re.search(r"(\d{4})[-_/](\d{1,2})\b", name)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return pd.Timestamp(y, mo, 1)

    # 3) Year + short month name (YYYY[-_/](Jan|Feb|...))
    m = re.search(r"(\d{4})[-_/](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", name, re.I)
    if m:
        y, mon = int(m.group(1)), m.group(2).title()
        mo = pd.to_datetime(mon, format="%b").month
        return pd.Timestamp(y, mo, 1)

    # 4) Short month name + year ((Jan|...)[-_/]YYYY)
    m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_/](\d{4})\b", name, re.I)
    if m:
        mon, y = m.group(1).title(), int(m.group(2))
        mo = pd.to_datetime(mon, format="%b").month
        return pd.Timestamp(y, mo, 1)

    # 5) Last-resort: let dateutil try the whole string
    try:
        dt = dparser.parse(name, dayfirst=True, fuzzy=True)
        return pd.Timestamp(dt.year, dt.month, 1)
    except Exception:
        return pd.NaT


def _read_csv_from_zip(zf, name) -> pd.DataFrame | None:
    # skip directories and macOS resource forks
    base = name.split("/")[-1]
    if (name.endswith("/") or name.startswith("__MACOSX/") or "/._" in name
        or base.startswith("._")):
        return None

    raw = zf.read(name)
    attempts = [
        ("utf-8", "strict"),
        ("utf-8-sig", "strict"),
        ("cp1252", "strict"),
        ("latin1", "strict"),
        ("utf-8", "replace"),
        ("cp1252", "replace"),
    ]
    last_err = None
    for enc, err_mode in attempts:
        try:
            text = raw.decode(enc, errors=err_mode)
            df = pd.read_csv(io.StringIO(text), engine="python", sep=None, on_bad_lines="skip")
            print(f"Read {name} with encoding={enc}, errors={err_mode}")
            return _standardise_cols(df)
        except Exception as e:
            last_err = e
            continue
    raise last_err

def _match_col(cols: list[str], patterns: list[str]) -> str | None:
    for p in patterns:
        if p.startswith("re:"):
            r = re.compile(p[3:], re.I)
            hits = [c for c in cols if r.search(c)]
            if hits:
                return hits[0]
        else:
            cand = re.sub(r"[^\w]+", "_", p.strip().lower())
            if cand in cols:
                return cand
    return None

def _apply_template(df: pd.DataFrame, table: str, filename: str) -> pd.DataFrame:
    """Rename columns to canonical names per template and add 'postcode' + 'month'."""
    tmpl = SCHEMA_TEMPLATES[table]
    df = _standardise_cols(df)
    cols = list(df.columns)
    rename_map = {}

    # required fields
    for canonical, patterns in tmpl["required"].items():
        col = _match_col(cols, patterns)
        if not col:
            raise SystemExit(
                f"[{table}] Missing required column '{canonical}' in {filename}. "
                f"Have: {cols[:20]} ... Looked for: {patterns}"
            )
        rename_map[col] = canonical

    # optional fields
    for canonical, patterns in tmpl.get("optional", {}).items():
        col = _match_col(cols, patterns)
        if col:
            rename_map[col] = canonical

    out = df.rename(columns=rename_map).copy()

    # Normalise postcode
    out["postcode"] = out["postcode"].astype(str).str.extract(r"(\d{4})", expand=False)

    # Month inference: prefer 'date', else 'period', else filename
    month = pd.Series(pd.NaT, index=out.index)
    if "date" in out:
        month = out["date"].map(_parse_month)
    if month.isna().all() and "period" in out:
        month = out["period"].map(_parse_month)
    if month.isna().all():
        fallback = _parse_month_from_filename(filename)
        month = pd.Series(fallback, index=out.index)
    out["month"] = month

    return out

def _select_files(names: list[str], patterns: list[str]) -> list[str]:
    return [n for n in names if any(re.search(p, n, re.I) for p in patterns)]

# ---------- main ----------
def process_latest_zip():
    zips = sorted(RAW_DIR.glob("wa_bonds_*.zip"))
    if not zips:
        raise SystemExit("No bonds ZIP found. Run fetch_bonds.py first.")
    latest = zips[-1]
    print(f"Processing {latest.name} ...")

    with zipfile.ZipFile(latest, "r") as zf:
        names = zf.namelist()

        # --- Lodgements ---
        lodg_files = _select_files(names, SCHEMA_TEMPLATES["lodgements"]["file_patterns"])
        lodg_dfs = []
        for fn in lodg_files:
            df = _read_csv_from_zip(zf, fn)
            if df is not None and not df.empty:
                lodg_dfs.append(_apply_template(df, "lodgements", fn))
        if lodg_dfs:
            lodg = pd.concat(lodg_dfs, ignore_index=True)
            # keep rows with usable keys
            lodg = lodg.dropna(subset=["month", "postcode", "weekly_rent"])
            lodg["weekly_rent"] = pd.to_numeric(lodg["weekly_rent"], errors="coerce")
            g = (
                lodg.groupby(["postcode", "month"], as_index=False)
                    .agg(median_rent=("weekly_rent", "median"),
                         p90_rent=("weekly_rent", lambda s: s.quantile(0.9)),
                         count_lodgements=("weekly_rent", "size"))
            )
        else:
            warnings.warn("No lodgement file(s) detected in ZIP.")
            g = pd.DataFrame(columns=["postcode","month","median_rent","p90_rent","count_lodgements"])

        # --- Disposals ---
        disp_files = _select_files(names, SCHEMA_TEMPLATES["disposals"]["file_patterns"])
        disp_dfs = []
        for fn in disp_files:
            df = _read_csv_from_zip(zf, fn)
            if df is not None and not df.empty:
                disp_dfs.append(_apply_template(df, "disposals", fn))
        if disp_dfs:
            disp = pd.concat(disp_dfs, ignore_index=True)
            disp = disp.dropna(subset=["month", "postcode", "days_held"])
            disp["days_held"] = pd.to_numeric(disp["days_held"], errors="coerce")
            d = (
                disp.groupby(["postcode", "month"], as_index=False)
                    .agg(mean_days_held=("days_held","mean"),
                         count_disposals=("days_held","size"))
            )
        else:
            warnings.warn("No disposal/refund file(s) detected in ZIP.")
            d = pd.DataFrame(columns=["postcode","month","mean_days_held","count_disposals"])

        # --- Bonds by Postcode (stock) ---
        stock_files = _select_files(names, SCHEMA_TEMPLATES["stock"]["file_patterns"])
        stock_dfs = []
        for fn in stock_files:
            df = _read_csv_from_zip(zf, fn)
            if df is not None and not df.empty:
                stock_dfs.append(_apply_template(df, "stock", fn))
        if stock_dfs:
            stock = pd.concat(stock_dfs, ignore_index=True)
            stock = stock.dropna(subset=["month", "postcode", "bonds_held"])
            stock["bonds_held"] = pd.to_numeric(stock["bonds_held"], errors="coerce")
            s = (
                stock.groupby(["postcode","month"], as_index=False)
                    .agg(stock_bonds=("bonds_held","max"))   # avoid double-counting
            )

        else:
            warnings.warn("No stock-by-postcode file(s) detected in ZIP.")
            s = pd.DataFrame(columns=["postcode","month","stock_bonds"])

    # --- Merge & engineer features ---
    panel = (g.merge(d, on=["postcode","month"], how="outer")
               .merge(s, on=["postcode","month"], how="outer"))

    stock = panel.get("stock_bonds", pd.Series(dtype=float)).fillna(0).astype(float)
    lodg  = panel.get("count_lodgements", pd.Series(dtype=float)).fillna(0).astype(float)
    disp  = panel.get("count_disposals", pd.Series(dtype=float)).fillna(0).astype(float)

    panel["churn_rate"] = (lodg + disp) / stock.clip(lower=1.0)        # avoid /0
    panel["net_stock_change"] = lodg - disp

    panel = panel.sort_values(["postcode","month"])
    panel["rent_momentum"] = (
        panel.groupby("postcode")["median_rent"]
            .pct_change(fill_method=None)   # future-proof
            .fillna(0.0)
    )

    def _clean_poa(s):
        x = re.sub(r"\D", "", str(s))
        if x == "872":  # fix common '872' -> '0872'
            x = "0872"
        return x if len(x) == 4 else None

    panel["postcode"] = panel["postcode"].map(_clean_poa)
    panel = panel.dropna(subset=["postcode"])


    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = STAGE_DIR / "bonds_panel_postcode.parquet"
    panel.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(panel):,} rows.")

if __name__ == "__main__":
    process_latest_zip()
