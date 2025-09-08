# WA Rental Forecast

**Bayesian nowcast + forecast of rental price pressure at SA2 level (Western Australia).**  
Outputs a Top‑20 list, an interactive map, and a spreadsheet of probabilities that **next month’s median rent rises by more than 2%** (default threshold), with automatic monthly back‑testing once new data lands.

**Live site:** https://wa-rental-forecast.netlify.app

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
- [Website (static site + Netlify)](#website-static-site--netlify)
- [Automation (cron / WSL)](#automation-cron--wsl)
- [Performance tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Roadmap / ideas](#roadmap--ideas)
- [License & acknowledgements](#license--acknowledgements)
 - [Time‑Split & Walk‑Forward](#time-split--walk-forward)

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
      ├─ outputs/tables/top20_pressure_risers.csv
      ├─ outputs/tables/price_pressure_forecast_sa2_latest[_named].xlsx
      ├─ outputs/reports/map_price_pressure.html
      ├─ outputs/figures/top20_pressure_risers.png
      ├─ outputs/evaluations/forecast_eval_summary.csv (once actuals exist)
      └─ outputs/figures/forecast_calibration_YYYY‑MM.png (once actuals exist)
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
  build_site.py               # Static site generator into docs/
  run_all.py                  # Orchestrates end‑to‑end run
  column_templates.py         # Column template utilities for ingestion
  config.py                   # Paths/thresholds/seeds; edit here

scripts/
  monthly_job.sh              # Activates venv, runs pipeline + evaluation, logs

docs/                         # (committed) Static site (index.html, data/, geojson)
data_raw/                     # (ignored) AHDAP ZIPs, ABS GeoPackage, ABS correspondence
data_stage/                   # (ignored) parquet intermediates
outputs/                      # (ignored) all generated artifacts
  tables/                     #   CSV/XLSX (Top‑20, full spreadsheets)
  figures/                    #   PNGs (bar charts, calibration)
  reports/                    #   Folium HTML map
  evaluations/                #   Evaluation CSVs (summary, details, calibration CSV)
  tests/                      #   Test artifacts (reserved)
  logs/                       #   Run logs (cron/test output)
```

> **Keep the repo small.** Don’t commit `data_*` or `outputs/`. The `.gitignore` below handles this.

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
pip install -r requirements.txt
```

### Option B — Conda/Mamba (faster PyMC sampling)
```bash
mamba create -n wa-rental-forecast python=3.12 pymc arviz pytensor              pandas numpy pyarrow geopandas pyogrio shapely folium branca matplotlib requests tqdm openpyxl
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
- **Map performance**:
  - `validate_and_report.py` simplifies geometries in meters for faster HTML.

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

> When everything is wired, outputs land under `outputs/`:
> - `outputs/tables/price_pressure_forecast_sa2_latest.xlsx`
> - `outputs/tables/price_pressure_forecast_sa2_latest_named.xlsx`  
> and after the next data drop: `outputs/evaluations/forecast_eval_*` files.

---

## Outputs

- `outputs/tables/top20_pressure_risers.csv` – Top‑20 SA2 codes by probability.  
- `outputs/tables/price_pressure_forecast_sa2_latest.xlsx` – All SA2s (code, month, probability).  
- `outputs/tables/price_pressure_forecast_sa2_latest_named.xlsx` – Same + **SA2 names**.  
- `outputs/reports/map_price_pressure.html` – Interactive SA2 map (fast, simplified geometries).  
- `outputs/figures/top20_pressure_risers.png` – Static bar chart (top‑20).  
- **After actuals exist** (next month):  
  - `outputs/evaluations/forecast_eval_summary.csv` – AUC, Brier, log‑loss by month.  
  - `outputs/evaluations/forecast_eval_details_YYYY‑MM.csv` – Row‑level truth vs prob.  
  - `outputs/evaluations/forecast_calibration_YYYY‑MM.csv` & `outputs/figures/forecast_calibration_YYYY‑MM.png`.

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

## Time‑Split & Walk‑Forward

This repo includes a helper to train on an early block of months and validate on later months using a strict split (no leakage), plus an optional walk‑forward variant that refits per target month.

Prerequisites: run up to SA2 staging so `data_stage/bonds_panel_sa2.parquet` covers the period of interest.

```
python -m src.fetch_bonds
python -m src.process_bonds
python -m src.map_poa_sa2
```

Example: Train on Jan–Apr, validate on Jun–Sep

1) Fixed split (single fit):
```
python -m src.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09
```
This:
- Fits the Negative Binomial nowcast on Jan–Apr and predicts out‑of‑sample `availability_rate` for the base months (May–Aug).
- Fits the logistic forecast on Jan–Apr and produces probabilities for the target months (Jun–Sep).
- Appends predictions to `data_stage/price_pressure_forecast_sa2_history.parquet`.

2) Walk‑forward (operational realism):
```
python -m src.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09 \
  --walk-forward
```
For each target month T in Jun..Sep, this refits:
- Nowcast on all data up to and including T−1 (to estimate availability for base month T−1);
- Logistic on months from Jan up to T−2 (so labels do not use information at T).

Evaluate just these months (once actuals exist for each):
```
python -m src.evaluate_forecasts --start 2025-06 --end 2025-09
```
Rebuild the static site (adds/updates docs/data/*.json):
```
python -m src.build_site
```

---

## Website (static site + Netlify)

The repo contains a generator that builds a static site into **`docs/`**:

```bash
python -m src.build_site
# Produces:
# docs/index.html
# docs/sa2_wa_simplified.geojson
# docs/data/months.json
# docs/data/YYYY-MM.json (per-month)
# docs/summary.json (metrics per month once actuals exist)
```

You can open locally:
```bash
python -m http.server --directory docs 8000
# then visit http://localhost:8000
```

**Deploy to Netlify (recommended, no server required):**
```bash
# one-time
npm i -g netlify-cli
netlify login
netlify link  # select your wa-rental-forecast site, or use: netlify link --id <API_ID>

# build + deploy
python -m src.build_site
netlify deploy --dir=docs --prod
```

Optional `docs/_headers` (cache hints):
```
/index.html
  Cache-Control: no-cache

/sa2_wa_simplified.geojson
  Cache-Control: public, max-age=31536000, immutable

/data/*
  Cache-Control: public, max-age=3600
```

---

## Automation (cron / WSL)

Use `scripts/monthly_job.sh` to run the full pipeline and publish the site. Example crontab (7:30 on the 5th each month):

```
30 7 5 * * /home/<you>/projects/wa_rental_pressure/scripts/monthly_job.sh
```

The script does:
1) Activate venv  
2) `src.run_all` → `src.evaluate_forecasts`  
3) `src.build_site`  
4) `netlify deploy --dir=docs --prod` (if the CLI is available)

Logs are written to `outputs/logs/run_all_<timestamp>.log`.

Cron tips:
- PATH: cron often lacks `/usr/local/bin` (where `netlify` lives). The script now exports a safe PATH; alternatively set in crontab: `PATH=/usr/local/bin:/usr/bin:/bin`.
- Netlify auth: ensure you’ve run `netlify login` under the same user so cron inherits credentials. For headless servers, set `NETLIFY_AUTH_TOKEN` in the crontab env.
- Debugging: check the latest `outputs/logs/run_all_*.log` for lines showing `which netlify` and the deploy command.

---

## Performance tips

- **Map speed**: geometries are simplified in meters for the folium map and for the static site; increase the tolerance for smaller files.  
- **Sampling speed**: consider a Conda/Mamba environment for optimized BLAS; or reduce `draws`/`tune` for dev runs.
- **Run‑time knobs** (safe defaults in code): `draws=1000`, `tune=1000`, `chains=4`, `target_accept=0.95`.

---

## Troubleshooting

**Crosswalk parse errors**  
- If the ABS correspondence is a **mega‑ZIP mis‑saved as .xlsx**, `map_poa_sa2.py` will detect it and open the inner file.  
- If CSV has weird delimiters/encodings, the loader **sniffs** them.  
- Script prints: `poa=... sa2=... ratio=... scale=ratio/percent rows=...`

**Timestamp not JSON serializable**  
- Fixed: the site builder strips/normalizes datetimes before serializing.

**Folium map feels slow**  
- Geometries are simplified in meters; raise tolerance for more speed/smaller HTML.

**PyMC `MutableData` missing / BLAS warning**  
- Not used: the models work with NumPy arrays (PyMC 5).  
- BLAS warning = performance only; Conda builds usually resolve it.

**Keep the repo small**  
See `.gitignore` below; do **not** commit `data_*` or anything in `outputs/`.

---

## Roadmap / ideas

- **Features**: `lodge_rate`, `stock_growth_1m`; seasonality in logistic.  
- **Pooling**: SA3 random intercepts above SA2.  
- **Calibration**: Platt/isotonic scaling if over/under‑confident.  
- **Notifications**: Slack/Discord webhook summary after each run.  
- **Distribution**: `environment.yml` for Conda users.

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
