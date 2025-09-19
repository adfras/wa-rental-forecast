#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git curl wget unzip zip \
    gdal-bin libgdal-dev libspatialindex-dev libproj-dev proj-bin libgeos-dev libgeos++-dev

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "Setup complete. Place ABS assets into data_raw/ and run: python -m src.cli one-click"
