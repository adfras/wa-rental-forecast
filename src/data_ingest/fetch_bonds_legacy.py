"""
Best-effort fetcher for legacy WA Rental Bonds ZIPs (pre‑2023).

This does not guess hidden endpoints. Instead it:
  - reads a manifest of URLs from env or a file and downloads them to
    data_raw/legacy/; and/or
  - copies any local ZIP paths provided.

Environment options:
  WA_BONDS_LEGACY_URLS     newline/space‑separated list of URLs or local paths
  WA_BONDS_LEGACY_MANIFEST path to a text file of URLs/paths (one per line)

CLI usage:
  python -m src.data_ingest.fetch_bonds_legacy \
      [--from-manifest scripts/wa_bonds_legacy_sources.txt] \
      [--url https://.../2019.zip] [--url file:///tmp/2020.zip] ...

After fetching, run:
  python -m src.data_ingest.process_bonds --all   # to combine latest + legacy
"""
from __future__ import annotations

from pathlib import Path
import argparse
import os
import shutil
import sys
import urllib.parse
import requests

from datetime import datetime
from src.config import RAW_DIR


def _iter_sources(env_urls: str | None, manifest_path: str | None, cli_urls: list[str]) -> list[str]:
    out: list[str] = []
    def add_line(s: str) -> None:
        s = s.strip()
        if not s or s.startswith('#'):
            return
        out.append(s)

    if env_urls:
        for tok in env_urls.replace("\r", "\n").split():
            add_line(tok)

    if manifest_path:
        p = Path(manifest_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                add_line(line)

    for u in cli_urls:
        add_line(u)

    # fallback: scripts/wa_bonds_legacy_sources.txt if present
    default_manifest = Path("scripts/wa_bonds_legacy_sources.txt")
    if default_manifest.exists():
        for line in default_manifest.read_text(encoding="utf-8").splitlines():
            add_line(line)

    # de‑duplicate while preserving order
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _download_to(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"", "file"}:
        # local file path
        src = Path(parsed.path if parsed.scheme == "file" else url).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Local file not found: {src}")
        target = dest_dir / src.name
        if src.resolve() == target.resolve():
            return target
        shutil.copy2(src, target)
        return target

    # http(s)
    name = Path(parsed.path).name or f"legacy_{abs(hash(url)) & 0xffffffff:x}.zip"
    if not name.lower().endswith(".zip"):
        name += ".zip"
    target = dest_dir / name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return target


def _try_head(url: str, timeout: float = 10.0) -> int | None:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code
    except Exception:
        return None


def _try_get(url: str, dest_dir: Path) -> Path | None:
    try:
        return _download_to(url, dest_dir)
    except Exception:
        return None


def _auto_discover_and_fetch(year_from: int, year_to: int, dest_dir: Path) -> list[Path]:
    """Best-effort attempts at known historical locations.

    1) AHDAP public S3 monthly zips if they exist by name pattern
       https://ahdap-public-data.s3.ap-southeast-2.amazonaws.com/RentalBondsWA/wa-rental-bond-jan2019.zip
    2) Old Commerce WA annual ZIPs via Wayback (if archived):
       bonds_by_postcode_summary_csvYYYY.zip, monthly_bond_lodgement_summary_csvYYYY.zip, monthly_bond_disposal_summary_csvYYYY.zip
    """
    fetched: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1) AHDAP S3 monthlies
    months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    for y in range(year_from, year_to + 1):
        for m in months:
            url = f"https://ahdap-public-data.s3.ap-southeast-2.amazonaws.com/RentalBondsWA/wa-rental-bond-{m}{y}.zip"
            code = _try_head(url)
            if code == 200:
                p = _try_get(url, dest_dir)
                if p:
                    print(f"OK  {url}")
                    fetched.append(p)

    # 2) Wayback for annual bundles on the old site
    def wayback_candidates(year: int) -> list[str]:
        base = "https://www.commerce.wa.gov.au/sites/default/files/atoms/files"
        names = [
            f"bonds_by_postcode_summary_csv{year}.zip",
            f"monthly_bond_lodgement_summary_csv{year}.zip",
            f"monthly_bond_disposal_summary_csv{year}.zip",
        ]
        return [f"{base}/{n}" for n in names]

    def cdx_lookup(u: str) -> str | None:
        api = (
            "https://web.archive.org/cdx/search/cdx?output=json&fl=timestamp,original,statuscode&filter=statuscode:200&limit=1&url="
            + requests.utils.quote(u, safe="")
        )
        try:
            j = requests.get(api, timeout=15).json()
            if isinstance(j, list) and len(j) > 1 and len(j[1]) >= 1:
                ts = j[1][0]
                return f"https://web.archive.org/web/{ts}id_/{u}"
        except Exception:
            return None
        return None

    for y in range(year_from, year_to + 1):
        for u in wayback_candidates(y):
            wb = cdx_lookup(u)
            if not wb:
                continue
            p = _try_get(wb, dest_dir)
            if p:
                print(f"OK  {wb}")
                fetched.append(p)

    return fetched


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch legacy WA bonds ZIPs (pre‑2023) into data_raw/legacy/")
    ap.add_argument("--from-manifest", dest="manifest", default=os.getenv("WA_BONDS_LEGACY_MANIFEST"))
    ap.add_argument("--url", dest="urls", action="append", default=[])
    ap.add_argument("--auto", nargs="?", const="2017:2022", help="Auto-discover common locations (default years 2017:2022); format: YYYY:YYYY")
    args = ap.parse_args(argv)

    out_dir = RAW_DIR / "legacy"
    fetched: list[Path] = []
    errors = 0

    # 1) Auto-discover if requested
    if args.auto:
        try:
            yfrom, yto = args.auto.split(":", 1)
            y1, y2 = int(yfrom), int(yto)
        except Exception:
            y2 = datetime.utcnow().year - 1
            y1 = max(2010, y2 - 8)
        fetched.extend(_auto_discover_and_fetch(y1, y2, out_dir))

    # 2) Manifest/URLs
    sources = _iter_sources(os.getenv("WA_BONDS_LEGACY_URLS"), args.manifest, args.urls)
    for src in sources:
        try:
            path = _download_to(src, out_dir)
            fetched.append(path)
            print(f"OK  {src} -> {path}")
        except Exception as e:
            errors += 1
            print(f"FAIL {src}: {e}", file=sys.stderr)

    if fetched and errors == 0:
        print(f"Fetched {len(fetched)} legacy ZIP(s) into {out_dir}")
    elif fetched:
        print(f"Fetched {len(fetched)} legacy ZIP(s) with {errors} error(s) into {out_dir}")
    else:
        print("No legacy ZIPs fetched.")


if __name__ == "__main__":
    main()
