"""
Backfill forecast history by merging predictions found in docs/data/YYYY-MM.json
into data_stage/price_pressure_forecast_sa2_history.parquet, alongside the latest
price_pressure_forecast_sa2.parquet if present.

This ensures month-by-month evaluations have predictions for past months even if
the Parquet history was missing when the site JSON already had those months.
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.config import STAGE_DIR


DOCS_DATA = Path("docs/data")


def _read_docs_predictions() -> pd.DataFrame:
    rows = []
    if not DOCS_DATA.exists():
        return pd.DataFrame(columns=["sa2_code", "month", "price_pressure_prob"]).astype({"sa2_code": str})
    for p in sorted(DOCS_DATA.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")):
        month_str = p.stem
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for sa2_code, rec in obj.items():
            pval = rec.get("p", None)
            rows.append({
                "sa2_code": str(sa2_code),
                "month": pd.to_datetime(month_str).to_period("M").to_timestamp(),
                "price_pressure_prob": (None if pval is None else float(pval)),
            })
    if not rows:
        return pd.DataFrame(columns=["sa2_code", "month", "price_pressure_prob"]).astype({"sa2_code": str})
    df = pd.DataFrame(rows)
    df["sa2_code"] = df["sa2_code"].astype(str)
    return df


def main():
    hist_path = STAGE_DIR / "price_pressure_forecast_sa2_history.parquet"
    latest_path = STAGE_DIR / "price_pressure_forecast_sa2.parquet"

    parts = []
    # Existing history first (preferred when duplicates exist)
    if hist_path.exists():
        try:
            parts.append(pd.read_parquet(hist_path))
        except Exception:
            pass
    # Latest predictions next
    if latest_path.exists():
        try:
            parts.append(pd.read_parquet(latest_path))
        except Exception:
            pass
    # Docs/site JSON last to fill any gaps
    docs_df = _read_docs_predictions()
    if not docs_df.empty:
        parts.append(docs_df)

    if not parts:
        print("No sources found to backfill history.")
        return

    df = pd.concat(parts, ignore_index=True)
    # Normalize dtypes
    df["sa2_code"] = df["sa2_code"].astype(str)
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    # Keep first occurrence (history > latest > docs)
    df = df.drop_duplicates(subset=["sa2_code", "month"], keep="first")
    df = df.sort_values(["sa2_code", "month"]).reset_index(drop=True)

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(hist_path, index=False)
    print(
        f"Backfilled forecast history → {hist_path} (months={df['month'].nunique()}, rows={len(df)})"
    )


if __name__ == "__main__":
    main()

