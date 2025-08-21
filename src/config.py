# Project configuration
# Source: Implemented from the WA Rental Pressure Forecaster design brief.
from pathlib import Path

# Base directories (computed relative to this file)
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data_raw"

STAGE_DIR = BASE_DIR / "data_stage"
OUT_DIR = BASE_DIR / "data_out"
FIG_DIR = BASE_DIR / "figures"
REPORT_DIR = BASE_DIR / "reports"

# External sources (adjust if portals change)
CKAN_BASE = "https://housing-data-exchange.ahdap.org"
CKAN_DATASET_NAME = "WA Rental Bonds Data 2023 - Current"

# ABS assets (place into data_raw/)
# - ASGS 2021 SA2 GeoPackage (often named like 'ASGS_2021_SA2.gpkg')
# - POA->SA2 2021 correspondence CSV (population allocation ratios)
ASGS_SA2_GPKG = RAW_DIR / "ASGS_2021_SA2.gpkg"
POA_SA2_CORRESP = RAW_DIR / "CG_POA_2021_SA2_2021.xlsx"


# Modelling constants
RENT_GROWTH_THRESHOLD = 0.02  # 2% month-over-month
RANDOM_SEED = 42

def ensure_dirs():
    for d in [RAW_DIR, STAGE_DIR, OUT_DIR, FIG_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
