"""Lightweight fetcher for REIWA suburb median house prices for K10/K20.

Reads action lists from outputs/tables/action_top10.csv and action_top20.csv,
maps SA2 names to REIWA suburb slugs with simple heuristics (and optional
overrides), fetches the suburb page, and tries to extract the latest median
house price and an updated date. Writes a consolidated CSV:

  outputs/tables/reiwa_topk_prices.csv

Notes
- This is a best-effort, low-volume fetch (<= a few pages per run). It should
  be rate-limited and resilient; failures are recorded with status.
- No bulk scraping; pages and selectors may change. If so, use overrides.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.config import OUT_DIR


BASE = "https://reiwa.com.au/suburb/"
HEADERS = {
    "User-Agent": "WA-Rental-Forecast/1.0 (contact: local use)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _read_action_lists(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["sa2_code", "sa2_name", "price_pressure_prob", "month"])  # type: ignore
    df = pd.concat(frames, ignore_index=True)
    # Ensure presence of names; caller writes sa2_name in action lists
    if "sa2_name" not in df.columns:
        df["sa2_name"] = None
    # Deduplicate by SA2
    df = df.sort_values(["price_pressure_prob"], ascending=False)
    df = df.drop_duplicates(subset=["sa2_code"]).reset_index(drop=True)
    return df


def _load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["sa2_code", "sa2_name", "suburb_slug", "url"])  # type: ignore
    try:
        df = pd.read_csv(path)
        for c in ["sa2_code", "sa2_name", "suburb_slug", "url"]:
            if c not in df.columns:
                df[c] = None
        df["sa2_code"] = df["sa2_code"].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["sa2_code", "sa2_name", "suburb_slug", "url"])  # type: ignore


def _slugify(s: str) -> str:
    # Remove parentheticals, trim, lower, hyphenate
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _candidate_slugs(name: str) -> list[str]:
    if not name:
        return []
    # Split on common multi-suburb separators
    parts = re.split(r"\s*-\s*|/", name)
    slugs = []
    for p in parts:
        slug = _slugify(p)
        if slug:
            slugs.append(slug)
    # Also try the whole name slug
    whole = _slugify(name)
    if whole and whole not in slugs:
        slugs.append(whole)
    return slugs


@dataclass
class FetchResult:
    sa2_code: str
    sa2_name: str | None
    suburb_slug: str
    source_url: str
    median_house_price: float | None
    updated_text: str | None
    fetched_at: str
    status: str


def _parse_reiwa(html: str) -> tuple[float | None, str | None]:
    # Try several robust patterns
    # 1) "Median house price $600,000" or similar
    for pat in [
        r"Median\s+house\s+price[^$]*\$([\d,]+)",
        r"Median\s+house\s+sale\s+price[^$]*\$([\d,]+)",
        r"Median\s+price[^$]*\$([\d,]+)\s*(?:house|Houses)?",
    ]:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                price = val
                break
            except Exception:
                price = None
                break
    else:
        price = None

    # Updated date text
    upd = None
    for up in [r"Last\s+updated[:\s]+([^<\n]+)", r"Updated[:\s]+([^<\n]+)"]:
        m2 = re.search(up, html, flags=re.IGNORECASE)
        if m2:
            upd = m2.group(1).strip()
            break
    return price, upd


def fetch_suburb(slug: str, *, session: requests.Session | None = None, timeout: float = 15.0) -> tuple[float | None, str | None, int]:
    url = BASE + slug + "/"
    sess = session or requests.Session()
    try:
        r = sess.get(url, headers=HEADERS, timeout=timeout)
        status = r.status_code
        if status == 200 and r.text:
            price, upd = _parse_reiwa(r.text)
            return price, upd, status
        return None, None, status
    except Exception:
        return None, None, -1


def main(sleep_sec: float = 1.0, max_pages: int = 40) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUT_DIR / "action_top10.csv", OUT_DIR / "action_top20.csv"]
    acts = _read_action_lists(paths)
    if acts.empty:
        print("No action lists found; skipping REIWA fetch.")
        return

    overrides = _load_overrides(Path("data_raw/reiwa_overrides.csv"))
    sess = requests.Session()
    rows: list[FetchResult] = []
    seen = set()
    pages = 0
    for _, row in acts.iterrows():
        sa2_code = str(row.get("sa2_code", ""))
        sa2_name = str(row.get("sa2_name", "")) if pd.notna(row.get("sa2_name", None)) else None
        if sa2_code in seen:
            continue
        seen.add(sa2_code)

        # Override first
        over = overrides[overrides["sa2_code"] == sa2_code]
        candidates: list[tuple[str, str]] = []  # (slug, url)
        if len(over) > 0:
            for _, ov in over.iterrows():
                slug = str(ov.get("suburb_slug", "")).strip()
                url = str(ov.get("url", "")).strip() if pd.notna(ov.get("url", None)) else ""
                if slug:
                    candidates.append((slug, url or BASE + slug + "/"))

        # Heuristic candidates from SA2 name
        if not candidates and sa2_name:
            for slug in _candidate_slugs(sa2_name):
                candidates.append((slug, BASE + slug + "/"))

        if not candidates:
            # Nothing to try
            rows.append(
                FetchResult(sa2_code, sa2_name, "", "", None, None, pd.Timestamp.utcnow().isoformat(), "no-candidate")
            )
            continue

        success = False
        for slug, url in candidates:
            if pages >= max_pages:
                break
            price, upd, status = fetch_suburb(slug, session=sess)
            pages += 1
            rows.append(
                FetchResult(
                    sa2_code=sa2_code,
                    sa2_name=sa2_name,
                    suburb_slug=slug,
                    source_url=url,
                    median_house_price=price,
                    updated_text=upd,
                    fetched_at=pd.Timestamp.utcnow().isoformat(),
                    status=str(status),
                )
            )
            if price is not None:
                success = True
                break
            time.sleep(max(sleep_sec, 0.0))

    out = pd.DataFrame([r.__dict__ for r in rows])
    # Keep best (first successful) per SA2
    # Use a helper column for availability of price
    out["_has_price"] = out["median_house_price"].notna()
    out = (
        out.sort_values(["sa2_code", "_has_price", "fetched_at"], ascending=[True, False, True])
        .drop_duplicates(subset=["sa2_code"], keep="first")
        .drop(columns=["_has_price"])
    )
    path = OUT_DIR / "reiwa_topk_prices.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
