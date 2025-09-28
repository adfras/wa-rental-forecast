"""Derive a WA suburb slug list from SAL→SA2 (2021) mapping.

Input:
  data_raw/ssc_to_sa2_2021.csv (built by build_sal_to_sa2_2021.py)

Output:
  data_raw/reiwa_suburbs.csv with columns: suburb_slug, suburb_name, state
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import re


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[–—/]", " ", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def main() -> None:
    src = Path("data_raw/ssc_to_sa2_2021.csv")
    if not src.exists():
        raise SystemExit("Missing data_raw/ssc_to_sa2_2021.csv. Run: python -m tools.build_sal_to_sa2_2021")
    m = pd.read_csv(src, dtype=str)
    suburbs = (
        m[["ssc_name"]]
        .drop_duplicates()
        .rename(columns={"ssc_name": "suburb_name"})
        .assign(suburb_name=lambda d: d["suburb_name"].fillna("").astype(str))
        .assign(suburb_slug=lambda d: d["suburb_name"].map(slugify), state="WA")
        [["suburb_slug", "suburb_name", "state"]]
        .sort_values("suburb_slug")
        .reset_index(drop=True)
    )
    out = Path("data_raw/reiwa_suburbs.csv")
    suburbs.to_csv(out, index=False)
    print(f"Wrote {out} ({suburbs.shape[0]} rows)")


if __name__ == "__main__":
    main()
