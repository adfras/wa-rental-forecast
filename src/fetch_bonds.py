import os, sys, time
import requests
from datetime import datetime
from src.config import CKAN_BASE, CKAN_DATASET_NAME, RAW_DIR

API = f"{CKAN_BASE}/api/3/action"

def _get(url, **params):
    with requests.Session() as s:
        r = s.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

def package_search(q):
    # Basic search by dataset name; adjust if your CKAN uses different fields
    res = _get(f"{API}/package_search", q=q)
    return res["result"]["results"]

def latest_resource_url(pkg):
    # Choose the latest ZIP resource by last_modified or created
    resources = [r for r in pkg.get("resources", []) if (r.get("format","") or "").lower() == "zip" or (r.get("url","").lower().endswith(".zip"))]
    if not resources:
        raise RuntimeError("No ZIP resources found in dataset.")
    def key(r):
        return r.get("last_modified") or r.get("created") or ""
    resources.sort(key=key, reverse=True)
    return resources[0]["url"]

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pkgs = package_search(CKAN_DATASET_NAME)
    if not pkgs:
        raise SystemExit("Dataset not found. Check CKAN_DATASET_NAME in src/config.py.")
    url = latest_resource_url(pkgs[0])
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"wa_bonds_{ts}.zip"
    print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
