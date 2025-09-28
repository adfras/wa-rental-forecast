"""Resumable, rate-limited crawler for REIWA suburb pages (with permission).

WARNING: Run only if you have explicit permission from REIWA to fetch all
suburb pages. This script is single-threaded, polite (sleep between requests),
and resumable. It stores raw HTML for each suburb so you can reparse later.

Inputs
  - data_raw/reiwa_suburbs.csv  (suburb_slug, suburb_name, state=WA)
  - data_raw/reiwa_overrides.csv (optional; columns: suburb_slug,url)

Outputs
  - outputs/reiwa_html/{slug}.html           (raw HTML per suburb)
  - outputs/tables/reiwa_crawl_log.csv       (per-attempt log)
  - outputs/tables/reiwa_medians_all.csv     (latest parsed medians per suburb)

Usage
  python -m tools.reiwa_crawl_all --sleep-lo 6 --sleep-hi 10 --resume
  # For a dry subset: --max-pages 50
"""
from __future__ import annotations

import argparse
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.config import OUT_DIR
from tqdm import tqdm


BASE = "https://reiwa.com.au/suburb/"
HEADERS = {
    "User-Agent": "WA-Rental-Forecast/1.0 (contact: internal research)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _read_suburbs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["state"].str.upper().eq("WA")]
    # Normalize slugs
    df["suburb_slug"] = df["suburb_slug"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["suburb_slug"]).reset_index(drop=True)
    return df


def _load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        out = {}
        for _, r in df.iterrows():
            slug = (r.get("suburb_slug", "") or "").strip().lower()
            url = (r.get("url", "") or "").strip()
            if slug and url:
                out[slug] = url
        return out
    except Exception:
        return {}


def _parse_reiwa(html: str) -> tuple[float | None, float | None, float | None, str | None]:
    # Regex-only parse to avoid extra deps. Be robust to $ separators and commas.
    def _find(pats: Iterable[str]) -> float | None:
        for pat in pats:
            m = re.search(pat, html, flags=re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except Exception:
                    return None
        return None

    p_median = _find([
        r"Median\s+house\s+price[^$]*\$\s*([\d,]+)",
        r"Median\s+house\s+sale\s+price[^$]*\$\s*([\d,]+)",
        r"Median\s+price[^$]*\$\s*([\d,]+)\b",
    ])
    p25 = _find([
        r"Lower\s+quartile[^$]*\$\s*([\d,]+)",
        r"25(?:th)?\s*percentile[^$]*\$\s*([\d,]+)",
    ])
    p75 = _find([
        r"Upper\s+quartile[^$]*\$\s*([\d,]+)",
        r"75(?:th)?\s*percentile[^$]*\$\s*([\d,]+)",
    ])
    upd = None
    for up in [r"Based\s+on\s+settled\s+sales[^<\n]+", r"Last\s+updated[:\s]+([^<\n]+)"]:
        m = re.search(up, html, flags=re.IGNORECASE)
        if m:
            upd = m.group(0) if m.lastindex is None else m.group(1)
            break
    return p25, p_median, p75, (upd.strip() if upd else None)


@dataclass
class CrawlRow:
    suburb_slug: str
    suburb_name: str
    url: str
    status: int
    fetched_at: str
    median_house_price: float | None
    p25_house_price: float | None
    p75_house_price: float | None
    last_updated_text: str | None


def main(
    *,
    resume: bool = True,
    sleep_lo: float = 6.0,
    sleep_hi: float = 10.0,
    max_pages: int | None = None,
    progress: bool = True,
) -> None:
    src = Path("data_raw/reiwa_suburbs.csv")
    if not src.exists():
        raise SystemExit("Missing data_raw/reiwa_suburbs.csv. Run: python -m src.cli build-reiwa-suburbs")
    subs = _read_suburbs(src)
    override = _load_overrides(Path("data_raw/reiwa_overrides.csv"))

    html_dir = OUT_DIR.parent / "reiwa_html"
    html_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "reiwa_crawl_log.csv"
    out_path = OUT_DIR / "reiwa_medians_all.csv"

    sess = requests.Session()
    rows: list[CrawlRow] = []
    # Prepare progress bar (start at number of already-fetched HTMLs when resuming)
    existing = 0
    if resume:
        existing = sum((html_dir / f"{slug}.html").exists() for slug in subs["suburb_slug"].astype(str))
    # Robust ASCII progress bar with counts and ETA; adapts to terminal width
    pbar = None
    parse_pbar = None
    if existing < len(subs):
        pbar = tqdm(
            total=len(subs),
            initial=existing,
            desc="REIWA fetch",
            unit="suburb",
            disable=not progress,
            dynamic_ncols=True,
            ascii=True,
            mininterval=0.5,
            maxinterval=2.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    else:
        # No fetch needed: show a separate parse progress bar for cached HTML
        if progress:
            parse_pbar = tqdm(
                total=len(subs),
                initial=0,
                desc="REIWA parse",
                unit="suburb",
                dynamic_ncols=True,
                ascii=True,
                mininterval=0.5,
                maxinterval=2.0,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

    done = 0
    for _, r in subs.iterrows():
        slug = r["suburb_slug"].strip().lower()
        name = r["suburb_name"]
        url = override.get(slug) or (BASE + slug + "/")
        html_file = html_dir / f"{slug}.html"

        if resume and html_file.exists():
            try:
                html = html_file.read_text(encoding="utf-8", errors="ignore")
                p25, median, p75, upd = _parse_reiwa(html)
                rows.append(CrawlRow(slug, name, url, 200, pd.Timestamp.utcnow().isoformat(), median, p25, p75, upd))
                done += 1
                if parse_pbar is not None:
                    parse_pbar.update(1); parse_pbar.refresh()
                if max_pages and done >= max_pages:
                    break
                if pbar is not None:
                    pbar.update(1); pbar.refresh()
                continue
            except Exception:
                pass

        try:
            resp = sess.get(url, headers=HEADERS, timeout=30)
            status = resp.status_code
            text = resp.text if status == 200 else ""
        except Exception:
            status = -1
            text = ""

        if status == 200 and text:
            try:
                html_file.write_text(text, encoding="utf-8")
            except Exception:
                pass
            p25, median, p75, upd = _parse_reiwa(text)
        else:
            p25 = median = p75 = None
            upd = None

        rows.append(CrawlRow(slug, name, url, status, pd.Timestamp.utcnow().isoformat(), median, p25, p75, upd))
        done += 1
        # Periodic flush
        if len(rows) % 50 == 0:
            pd.DataFrame([r.__dict__ for r in rows]).to_csv(log_path, index=False)
        if max_pages and done >= max_pages:
            break
        # Sleep politely between requests
        time.sleep(random.uniform(max(sleep_lo, 0.0), max(sleep_hi, sleep_lo)))
        if parse_pbar is not None:
            parse_pbar.update(1); parse_pbar.refresh()
        if pbar is not None:
            pbar.update(1); pbar.refresh()

    if pbar is not None:
        pbar.close()
    if parse_pbar is not None:
        parse_pbar.close()
    log_df = pd.DataFrame([r.__dict__ for r in rows])
    log_df.to_csv(log_path, index=False)
    # Keep only latest row per suburb (prefer successful parse)
    if not log_df.empty:
        best = (
            log_df.assign(_has_price=log_df["median_house_price"].notna())
            .sort_values(["_has_price", "fetched_at"], ascending=[False, True])
            .drop(columns=["_has_price"])
            .drop_duplicates(subset=["suburb_slug"], keep="first")
            .sort_values("suburb_slug")
        )
        best.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({best.shape[0]} rows)")
    print(f"Crawled {done} suburb pages (resume={resume}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crawl REIWA suburb pages (with permission)")
    ap.add_argument("--sleep-lo", type=float, default=6.0)
    ap.add_argument("--sleep-hi", type=float, default=10.0)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--resume", dest="resume", action="store_true")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    ap.set_defaults(resume=True)
    args = ap.parse_args()
    main(resume=args.resume, sleep_lo=args.sleep_lo, sleep_hi=args.sleep_hi,
         max_pages=args.max_pages, progress=args.progress)
