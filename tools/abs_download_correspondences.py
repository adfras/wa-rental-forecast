"""Download and unpack ASGS 2021 correspondences (Edition 3).

Writes:
  - data_raw/asgs2021_correspondences.zip
  - data_raw/asgs2021_correspondences/ (unpacked CSVs)

This contains CG_SAL_2021_SA2_2021.csv which maps SAL→SA2 with RATIO_FROM_TO.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import requests

URL = (
    "https://data.gov.au/data/dataset/2c79581f-600e-4560-80a8-98adb1922dfc/"
    "resource/33d822ba-138e-47ae-a15f-460279c3acc3/download/asgs2021_correspondences.zip"
)


def main() -> None:
    raw = Path("data_raw"); raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "asgs2021_correspondences.zip"
    out_dir = raw / "asgs2021_correspondences"

    print(f"Downloading ASGS 2021 correspondences → {zip_path}")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    zip_path.write_bytes(r.content)
    print(f"Saved {zip_path} ({len(r.content)} bytes)")

    print(f"Unpacking → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()

