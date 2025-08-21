# WA Rental Forecast

**Bayesian nowcast + forecast of rental price pressure at SA2 level (Western Australia).**  
Outputs a Top‑20 list, an interactive map, and a spreadsheet of probabilities that **next month’s median rent rises by more than 2%** (default threshold), with automatic monthly back‑testing once new data lands.

---

## Table of contents

- [Project goals](#project-goals)
- [Pipeline overview](#pipeline-overview)
- [Repository layout](#repository-layout)
- [Data sources](#data-sources)
- [Install & setup](#install--setup)
- [Configuration](#configuration)
- [How to run](#how-to-run)
- [Outputs](#outputs)
- [Models](#models)
- [Evaluation & back‑testing](#evaluation--back-testing)
- [Automation (cron / WSL)](#automation-cron--wsl)
- [Performance tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Roadmap / ideas](#roadmap--ideas)
- [License & acknowledgements](#license--acknowledgements)

---

## Project goals

- **Nowcast**: estimate current **rental availability** (tightness) for each SA2 in WA.  
- **Forecast**: probability that **next month’s median rent** increases by **> 2%** (configurable).

Deliverables each month:
- **Top‑20 risers CSV**
- **Interactive SA2 map** (folium HTML)
- **Static bar chart PNG**
- **Spreadsheet of all SA2 probabilities** (with SA2 names)
- **Evaluation files** once the next month’s data arrives (AUC/Brier/Calibration)

---

## Pipeline overview

```
AHDAP Bonds ZIPs ─┐
                   ├─> fetch_bonds.py ─> process_bonds.py ──┐
ABS POA→SA2 xwalk ─┘                                        │
ABS SA2 GeoPackage ─────────────────────────────────────────┘
                         ↓
           data_stage/bonds_panel_postcode.parquet
                         ↓
           map_poa_sa2.py  (POA→SA2, WA filter)
                         ↓
           data_stage/bonds_panel_sa2.parquet
                         ↓
           model_nowcast.py  (NB w/ exposure)
                         ↓
   data_stage/availability_nowcast_sa2.parquet
                         ↓
           model_forecast.py (hier. logistic)
                         ↓
   data_stage/price_pressure_forecast_sa2.parquet
                         ↓
     validate_and_report.py + evaluate_forecasts.py
      ├─ data_out/top20_pressure_risers.csv
      ├─ data_out/price_pressure_forecast_sa2_latest[_named].xlsx
      ├─ reports/map_price_pressure.html
      ├─ figures/top20_pressure_risers.png
      ├─ data_out/forecast_eval_summary.csv (once actuals exist)
      └─ figures/forecast_calibration_YYYY‑MM.png (once actuals exist)
```

---

## Repository layout

```
src/
  fetch_bonds.py              # Download latest WA bonds ZIP (AHDAP)
  process_bonds.py            # Ingest ZIPs → tidy postcode×month panel
  map_poa_sa2.py              # POA→SA2 mapping (robust to CSV/XLSX/ABS mega‑ZIP)
  model_nowcast.py            # PyMC Negative Binomial nowcast (exposure=stock)
  model_forecast.py           # PyMC hierarchical logistic forecast
  validate_and_report.py      # Top‑20 CSV, folium map, PNG bar chart (fast)
  evaluate_forecasts.py       # Back‑testing + named Excel export
  run_all.py                  # Orchestrates end‑to‑end run
  column_templates.py         # Column template utilities for ingestion
  config.py                   # Paths/thresholds/seeds; edit here

scripts/
  monthly_job.sh              # Activates venv, runs pipeline + evaluation, logs

data_raw/     # (not committed) AHDAP ZIPs, ABS GeoPackage, ABS correspondence
data_stage/   # (not committed) parquet intermediates
data_out/     # (not committed) CSV/XLSX outputs
figures/      # (not committed) PNGs
reports/      # (not committed) HTML map
logs/         # (not committed) run logs
```

> **Keep the repo small.** Don’t commit `data_*`, `reports/`, `figures/`, `logs/`. See `.gitignore` in [Troubleshooting](#troubleshooting).

---

## Data sources

- **WA Rental Bonds** (AHDAP CKAN): monthly lodgements, disposals, bonds held.  
- **ABS POA→SA2 correspondence (2021)**: population‑weighted allocation ratios.  
- **ABS ASGS 2021 SA2 GeoPackage**: geometries & names for SA2s.

Place these in `data_raw/`:

```
data_raw/
  <AHDAP_SLUGS.../ZIPs...>
  ASGS_2021_SA2.gpkg
  CG_POA_2021_SA2_2021.xlsx   # (or the ABS mega‑ZIP containing it)
```

`map_poa_sa2.py` is robust to:
- CSV or Excel,
- **or** the **ABS mega‑ZIP** mis‑saved as `.xlsx` (common!) — it will open the inner correspondence.

---

## Install & setup

### Option A — Python venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Core packages
pip install pandas numpy pyarrow geopandas pyogrio shapely folium branca \
            pymc arviz pytensor matplotlib requests tqdm
```

### Option B — Conda/Mamba (faster PyMC sampling)
```bash
mamba create -n wa-rental-forecast python=3.12 pymc arviz pytensor \
             pandas numpy pyarrow geopandas pyogrio shapely folium branca matplotlib
mamba activate wa-rental-forecast
```

> If you see a **BLAS** warning from PyTensor, Conda/Mamba builds usually fix it automatically.

---

## Configuration

All paths & constants live in `src/config.py`. Key settings:

- **Directories**: `RAW_DIR`, `STAGE_DIR`, `OUT_DIR`, `FIG_DIR`, `REPORT_DIR`  
- **ABS assets**:
  - `ASGS_SA2_GPKG = data_raw/ASGS_2021_SA2.gpkg`
  - `POA_SA2_CORRESP = data_raw/CG_POA_2021_SA2_2021.xlsx`
- **Model**:
  - `RENT_GROWTH_THRESHOLD = 0.02`  *(2% default threshold for a “rise”)*
  - `RANDOM_SEED = 42`
- **Map performance** (in `validate_and_report.py`):
  - `SIMPLIFY_TOL_M = 150` meters (raise to 300–500 for smaller/faster HTML)

---

## How to run

### End‑to‑end (one‑off)
```bash
# from repo root
python -m src.run_all
python -m src.validate_and_report
python -m src.evaluate_forecasts  # writes spreadsheets; scores if actuals exist
```

### Individual steps (iterate)
```bash
python -m src.fetch_bonds
python -m src.process_bonds
python -m src.map_poa_sa2
python -m src.model_nowcast
python -m src.model_forecast
python -m src.validate_and_report
python -m src.evaluate_forecasts
```

> When everything is wired, `evaluate_forecasts.py` produces:
> - `data_out/price_pressure_forecast_sa2_latest.xlsx`
> - `data_out/price_pressure_forecast_sa2_latest_named.xlsx`  
> and after the next data drop: `forecast_eval_*` files.

---

## Outputs

- `data_out/top20_pressure_risers.csv` – Top‑20 SA2 codes by probability.  
- `data_out/price_pressure_forecast_sa2_latest.xlsx` – All SA2s (code, month, probability).  
- `data_out/price_pressure_forecast_sa2_latest_named.xlsx` – Same + **SA2 names**.  
- `reports/map_price_pressure.html` – Interactive SA2 map (fast, simplified geometries).  
- `figures/top20_pressure_risers.png` – Static bar chart (top‑20).  
- **After actuals exist** (next month):  
  - `data_out/forecast_eval_summary.csv` – AUC, Brier, log‑loss by month.  
  - `data_out/forecast_eval_details_YYYY‑MM.csv` – Row‑level truth vs prob.  
  - `data_out/forecast_calibration_YYYY‑MM.csv` & `figures/forecast_calibration_YYYY‑MM.png`.

**Interpretation**: A value of **0.82** means an **82%** chance that next month’s median rent for that SA2 rises by **> 2%**.

---

## Models

### Nowcast (`src/model_nowcast.py`)
- **Likelihood**: Negative Binomial over **disposals** (ended tenancies).  
- **Offset**: `log(stock_bonds)` (exposure), so the expected count scales with stock.  
- **Structure**:
  - SA2 random intercepts (hierarchical pooling)
  - Linear time trend
  - Month‑of‑year seasonality (sum‑to‑zero)
  - Overdispersion parameter `alpha_nb`
- **Output**: `availability_rate` = posterior mean of expected count per unit stock.

### Forecast (`src/model_forecast.py`)
- **Target**: `1{ Δ median_rent_{t+1} > g }`, where `g = 2%` (configurable).  
- **Predictors** (standardized):
  - `availability_rate` (from nowcast)
  - `churn_rate` (disposals / stock)
  - `rent_mom_1m` and `rent_mom_3m` (momentum)  
- **Structure**:
  - Logistic regression with **SA2 random intercepts**
- **Output**: `price_pressure_prob` for each SA2 (latest month).

> **Extensions** (easy, optional): add `lodge_rate` and `stock_growth_1m` as extra features.

---

## Evaluation & back‑testing

`src/evaluate_forecasts.py` automatically:
- Builds **realized labels** when the next month’s SA2 panel is available.
- Scores predictions month‑by‑month: **AUC**, **Brier**, **log‑loss**, precision/recall at 0.5.
- Writes a **calibration plot** (predicted vs observed probability).

Until the next month’s panel exists, it will print:  
**“No realized months yet to score.”** (expected)

---

## Automation (cron / WSL)

The repo includes `scripts/monthly_job.sh` which:
- activates your venv
- runs `src.run_all` + `src.evaluate_forecasts`
- logs to `logs/run_all_<timestamp>.log`

**Set a monthly cron (WSL/Ubuntu):**
```bash
crontab -e
# 07:30 on the 5th of every month:
30 7 5 * * /home/<you>/projects/wa_rental_forecast/scripts/monthly_job.sh
```

**Test manually anytime:**
```bash
./scripts/monthly_job.sh
```

> Optional: add a Slack/Discord webhook to the script to ping on completion.

---

## Performance tips

- **Map speed**: geometries are simplified in meters (`SIMPLIFY_TOL_M`), properties limited to code/name/prob.  
  Bump to **300–500 m** if you want a much smaller HTML.
- **Sampling speed**: consider a Conda/Mamba environment for optimized BLAS; or reduce `draws`/`tune` for dev runs.
- **Run‑time knobs** (safe defaults in code): `draws=1000`, `tune=1000`, `chains=4`, `target_accept=0.95`.

---

## Troubleshooting

**Crosswalk parse errors**  
- If the ABS correspondence is a **mega‑ZIP mis‑saved as .xlsx**, `map_poa_sa2.py` will detect it and open the inner file.  
- If CSV has weird delimiters/encodings, the loader **sniffs** them.  
- Script prints: `poa=... sa2=... ratio=... scale=ratio/percent rows=...`

**Timestamp not JSON serializable**  
- Fixed: the map strips datetime fields before GeoJSON, only keeps minimal properties.

**Folium map feels slow**  
- We simplify geometries in meters; raise `SIMPLIFY_TOL_M` for more speed/smaller HTML.

**PyMC `MutableData` missing / BLAS warning**  
- Not used: the models work with NumPy arrays (PyMC3+ compatible).  
- BLAS warning = performance only; Conda builds usually resolve it.

**Keep the repo small**  
Create `.gitignore` in repo root:
```gitignore
/data_raw/
/data_stage/
/data_out/
/reports/
/figures/
/logs/
__pycache__/
*.pyc
*.pyo
```

---

## Roadmap / ideas

- **Features**: `lodge_rate`, `stock_growth_1m`; seasonality in logistic.  
- **Pooling**: SA3 random intercepts above SA2.  
- **Calibration**: Platt/isotonic scaling if over/under‑confident.  
- **Notifications**: Slack/Discord webhook summary after each run.  
- **Distribution**: simple `Makefile` and/or `environment.yml`.

---

## License & acknowledgements

- **Data**:  
  - WA rental bonds via AHDAP Housing Data Exchange.  
  - ABS ASGS 2021 SA2 geometries & POA→SA2 correspondence.  
  Respect the licensing of each dataset when sharing outputs.

- **Code**: choose a LICENSE (e.g., MIT) and place it in the repo root.

---

### One‑liner summary

> **WA Rental Forecast** produces a monthly, SA2‑level map and ranked list of **where rents are most likely to jump next month**, then automatically checks how accurate it was when the next data drop arrives.
