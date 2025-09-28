"""Create a URL manifest for all WA suburb pages on REIWA.

Input: data_raw/reiwa_suburbs.csv (suburb_slug, suburb_name, state)
Output: outputs/tables/reiwa_suburb_urls.csv with columns:
  suburb_slug, suburb_name, state, url

This is a safe, non-scraping manifest you can use for manual review.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import OUT_DIR


def main() -> None:
    src = Path("data_raw/reiwa_suburbs.csv")
    if not src.exists():
        raise SystemExit("Missing data_raw/reiwa_suburbs.csv. Run: python -m src.cli build-reiwa-suburbs")
    df = pd.read_csv(src, dtype=str).fillna("")
    df["url"] = "https://reiwa.com.au/suburb/" + df["suburb_slug"].str.strip().str.lower() + "/"
    df = df[["suburb_slug","suburb_name","state","url"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "reiwa_suburb_urls.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

