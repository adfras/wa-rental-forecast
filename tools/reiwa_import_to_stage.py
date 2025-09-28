"""Normalize suburb-level REIWA house prices into stage files.

Supports two data sources:
  1. Preferred: data_raw/WA_suburb_monthly_medians.xlsx (monthly timeseries)
  2. Legacy fallback: outputs/tables/reiwa_medians_all.csv (crawler snapshot)

Outputs (when the monthly workbook is present)
  - data_stage/house_prices_sal_monthly.parquet
      suburb_slug, suburb_name, month, period_span, median_house_price
  - data_stage/house_prices_sa2_monthly.parquet
      sa2_code, month, period_span, median_house_price, allocation_ratio
  - data_stage/house_prices_sal_snapshot.parquet
  - data_stage/house_prices_sa2_snapshot.parquet

When only the legacy CSV is available, the snapshot files are produced as before.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import STAGE_DIR

EXCEL_PATH = Path("data_raw/WA_suburb_monthly_medians.xlsx")
LEGACY_CSV_PATH = Path("outputs/tables/reiwa_medians_all.csv")
MAP_PATH = Path("data_raw/ssc_to_sa2_2021.csv")


def _slugify(s: str | float | None) -> str:
    import re as _re

    if s is None or (isinstance(s, float) and pd.isna(s)):
        base = ""
    else:
        base = str(s)
    base = base.strip().lower()
    base = _re.sub(r"\s*\(wa\)$", "", base, flags=_re.IGNORECASE)
    base = _re.sub(r"[–—/]", " ", base)
    base = _re.sub(r"[^a-z0-9]+", "-", base)
    base = _re.sub(r"-+", "-", base).strip("-")
    return base


def _load_mapping() -> pd.DataFrame:
    if not MAP_PATH.exists():
        raise SystemExit("Missing data_raw/ssc_to_sa2_2021.csv (SAL→SA2 mapping).")
    m = pd.read_csv(MAP_PATH, dtype=str)
    m["ssc_name"] = m["ssc_name"].astype(str)
    m["suburb_slug"] = m["ssc_name"].map(_slugify)
    m["allocation_ratio"] = pd.to_numeric(m["allocation_ratio"], errors="coerce").fillna(1.0)
    return m[["suburb_slug", "sa2_code", "allocation_ratio"]]


def _write_snapshot_files(sal: pd.DataFrame, sa2: pd.DataFrame, *, period_span: str) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    sal_snapshot = sal.sort_values(["suburb_slug", "month"]).groupby("suburb_slug", as_index=False).tail(1)
    sal_snapshot = sal_snapshot.assign(period_span=period_span)
    sal_path = STAGE_DIR / "house_prices_sal_snapshot.parquet"
    sal_snapshot.to_parquet(sal_path, index=False)
    print(f"Wrote {sal_path} ({len(sal_snapshot)} rows)")

    sa2_snapshot = sa2.sort_values(["sa2_code", "month"]).groupby("sa2_code", as_index=False).tail(1)
    sa2_snapshot = sa2_snapshot.assign(period_span=period_span)
    sa2_path = STAGE_DIR / "house_prices_sa2_snapshot.parquet"
    sa2_snapshot.to_parquet(sa2_path, index=False)
    print(f"Wrote {sa2_path} ({len(sa2_snapshot)} rows)")


def _process_monthly_workbook(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    print(f"Reading monthly workbook {path}")
    monthly = pd.read_excel(path, sheet_name="Monthly_Medians")
    monthly = monthly.rename(columns={"Suburb": "suburb_name"})
    monthly["suburb_name"] = monthly["suburb_name"].astype(str)

    # Filter out suburbs with population < 5k (if we can load the metadata sheet)
    eligible_slugs: set[str] | None = None
    try:
        pop = pd.read_excel(path, sheet_name="Suburbs_(>=5k_SAL)")
        pop["suburb_slug"] = pop["Suburb"].map(_slugify)
        pop["People_2021"] = pd.to_numeric(pop.get("People_2021"), errors="coerce")
        eligible_slugs = set(pop.loc[pop["People_2021"] >= 5000, "suburb_slug"].dropna().astype(str))
    except Exception:
        eligible_slugs = None

    value_cols: Iterable[str] = [c for c in monthly.columns if c != "suburb_name"]

    long = monthly.melt(id_vars="suburb_name", value_vars=value_cols,
                        var_name="month", value_name="median_house_price")
    long["month"] = pd.to_datetime(long["month"], format="%Y-%m", errors="coerce")
    long = long.dropna(subset=["month"]).copy()
    long["month"] = long["month"].dt.to_period("M").dt.to_timestamp()
    long["median_house_price"] = pd.to_numeric(long["median_house_price"], errors="coerce")
    long["suburb_slug"] = long["suburb_name"].map(_slugify)
    if eligible_slugs is not None and eligible_slugs:
        mask = long["suburb_slug"].isin(eligible_slugs)
        dropped_suburbs = sorted(set(long.loc[~mask, "suburb_slug"]))
        if dropped_suburbs:
            print(
                f"Filtered out {len(dropped_suburbs)} low-population suburbs (<5000 people) "
                "before SA2 aggregation."
            )
        long = long.loc[mask].copy()
    long = long.dropna(subset=["suburb_slug", "median_house_price"])
    long["period_span"] = "monthly_median"

    mapping = _load_mapping()
    sal = long[["suburb_slug", "suburb_name", "month", "period_span", "median_house_price"]].copy()
    sal = sal.sort_values(["suburb_slug", "month"]).reset_index(drop=True)

    sa2_raw = sal.merge(mapping, on="suburb_slug", how="left")
    missing = sa2_raw["sa2_code"].isna()
    if missing.any():
        missing_suburbs = sorted(sa2_raw.loc[missing, "suburb_name"].unique())
        if missing_suburbs:
            print(f"Warning: {len(missing_suburbs)} suburbs missing SA2 mapping; dropping. "
                  f"Examples: {missing_suburbs[:5]}")
    sa2_raw = sa2_raw.dropna(subset=["sa2_code"]).copy()
    sa2_raw["sa2_code"] = sa2_raw["sa2_code"].astype(str)

    def _collapse(group: pd.DataFrame) -> pd.Series:
        weights = group["allocation_ratio"].astype(float).fillna(0.0)
        vals = group["median_house_price"].astype(float)
        wsum = float(weights.sum())
        if wsum > 0:
            w_avg = float((vals * weights).sum() / wsum)
        else:
            w_avg = float(vals.mean()) if len(vals) else float("nan")
        return pd.Series({
            "median_house_price": w_avg,
            "period_span": group["period_span"].iloc[0],
            "allocation_weight_sum": wsum,
            "n_suburbs": int(len(group)),
        })

    sa2 = (
        sa2_raw.groupby(["sa2_code", "month"])
        .apply(_collapse, include_groups=False)
        .reset_index()
    )

    sal_monthly_path = STAGE_DIR / "house_prices_sal_monthly.parquet"
    sa2_monthly_path = STAGE_DIR / "house_prices_sa2_monthly.parquet"
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    sal.to_parquet(sal_monthly_path, index=False)
    sa2.to_parquet(sa2_monthly_path, index=False)
    print(f"Wrote {sal_monthly_path} ({len(sal)} rows)")
    print(f"Wrote {sa2_monthly_path} ({len(sa2)} rows)")

    _write_snapshot_files(sal, sa2, period_span="monthly_median")


def _parse_period_end(text: str | None, fetched_at: str) -> pd.Timestamp:
    # Try patterns like "Based on settled sales as at 20 September 2025"
    if text:
        match = re.search(r"as\s+at\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", text, flags=re.IGNORECASE)
        if match:
            try:
                dt = pd.to_datetime(match.group(1), dayfirst=True)
                return pd.Timestamp(dt.year, dt.month, 1)
            except Exception:
                pass
        alt = re.search(r"([A-Za-z]+\s+\d{4})", text)
        if alt:
            try:
                dt = pd.to_datetime("1 " + alt.group(1))
                return pd.Timestamp(dt.year, dt.month, 1)
            except Exception:
                pass
    # Fallback: month of fetch timestamp
    dtf = pd.to_datetime(fetched_at)
    return pd.Timestamp(dtf.year, dtf.month, 1)


def _process_legacy_csv(path: Path) -> None:
    if not path.exists():
        raise SystemExit("Missing outputs/tables/reiwa_medians_all.csv. Run the crawler first.")
    df = pd.read_csv(path)
    for c in ["suburb_slug", "suburb_name", "median_house_price", "p25_house_price", "p75_house_price", "last_updated_text", "fetched_at"]:
        if c not in df.columns:
            df[c] = None
    df["suburb_slug"] = df["suburb_slug"].astype(str).str.strip().str.lower()
    df["suburb_name"] = df["suburb_name"].astype(str)
    df["month"] = [
        _parse_period_end(text if isinstance(text, str) else None, str(fa))
        for text, fa in zip(df["last_updated_text"], df["fetched_at"])
    ]
    df["period_span"] = "12m_rolling"
    sal = df[[
        "suburb_slug", "suburb_name", "month", "period_span",
        "median_house_price", "p25_house_price", "p75_house_price"
    ]].copy()
    sal = sal.dropna(subset=["suburb_slug"]).drop_duplicates()

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    sal_path = STAGE_DIR / "house_prices_sal_snapshot.parquet"
    sal.to_parquet(sal_path, index=False)
    print(f"Wrote {sal_path} ({len(sal)} rows)")

    mapping = _load_mapping()
    j = sal.merge(mapping, on="suburb_slug", how="left")
    sa2 = j.dropna(subset=["sa2_code"])[["sa2_code", "month", "period_span", "median_house_price", "allocation_ratio"]].copy()
    sa2_path = STAGE_DIR / "house_prices_sa2_snapshot.parquet"
    sa2.to_parquet(sa2_path, index=False)
    print(f"Wrote {sa2_path} ({len(sa2)} rows)")


def main() -> None:
    if EXCEL_PATH.exists():
        _process_monthly_workbook(EXCEL_PATH)
    else:
        _process_legacy_csv(LEGACY_CSV_PATH)


if __name__ == "__main__":
    main()
