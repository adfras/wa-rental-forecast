# WA Rental Forecast

**Bayesian nowcast + forecast of rental price pressure at SA2 level (Western Australia).**  
Outputs a Top‑20 list, an interactive map, and a spreadsheet of probabilities that **next month’s median rent rises by more than 2%** (default threshold), with automatic monthly back‑testing once new data lands.

**Live site:** https://wa-rental-forecast.netlify.app

---

## Quick Start

```bash
# 1) Fit latest forecast (recency=6, calibrated, prior-shift, bias-correct)
make fit_latest

# 2) Evaluate recent months (example: last two)
make eval_recent START=2025-07 END=2025-08

# 3) Build and serve the site
make site
make serve   # open http://localhost:8081
```

> **New (Sept 2025)** – Major pipeline tidy-up:
> - Shared feature helpers now live under `src/features/`, keeping the forecast, validator, and utilities aligned on WA-demeaned drivers and interaction terms.
> - `src/models/forecast.py` now auto-calibrates, emits full threshold sweeps to `outputs/evaluations/forecast_training_metrics.json`, and supports optional prior-shift + bias fixes.
> - `src/data_ingest/external_signals.py` fetches RBA cash rates plus ABS unemployment/building approvals (via `curl`) so fresh macro signals land in `data_stage/external_signals.parquet` with lags for production.
> - Validation/run utilities moved under `tools/` (`time_split_validate`, `backfill_forecast_history`, etc.) and the docs site JSON is regenerated monthly (`docs/data/2025-01.json`…`2025-09.json`).

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
- [External signals (APIs)](#external-signals-apis)

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
           data_ingest/map_poa_sa2.py  (POA→SA2, WA filter)
                         ↓
           data_stage/bonds_panel_sa2.parquet
                         ↓
           models/nowcast.py  (NB w/ exposure)
                         ↓
   data_stage/availability_nowcast_sa2.parquet
                         ↓
           models/forecast.py (hier. logistic)
                         ↓
   data_stage/price_pressure_forecast_sa2.parquet
                         ↓
      reporting/validate_and_report.py + reporting/evaluate_forecasts.py
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
  config.py                   # Paths/thresholds/seeds; edit here
  column_templates.py         # Column template utilities for ingestion
  common/                     # Shared spatial + metadata helpers
  data_ingest/                # Raw data acquisition + staging
    fetch_bonds.py
    process_bonds.py
    map_poa_sa2.py
    external_signals.py
  features/
    dates.py
    engineering.py
  models/
    nowcast_design.py
    nowcast.py
    forecast.py
  reporting/
    validate_and_report.py
    evaluate_forecasts.py
    build_site.py
  cli/
    __main__.py               # `python -m src.cli <command>` dispatcher
    commands/one_click.py
    commands/run_all.py

scripts/
  monthly_job.sh              # Activates venv, runs pipeline + evaluation, logs

tools/                        # Diagnostics & utilities (time_split_validate, etc.)
docs/                         # (committed) Static site (index.html, data/, geojson)
data_raw/                     # (ignored) AHDAP ZIPs, ABS GeoPackage, ABS correspondence
data_stage/                   # (ignored) parquet intermediates
outputs/                      # (ignored) generated artifacts (tables/figures/reports/...)
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

`data_ingest/map_poa_sa2.py` is robust to:
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

## Lean Core vs Optional

- Core (src/): data_ingest/process_bonds → data_ingest/map_poa_sa2 → models/nowcast → models/forecast → reporting/validate_and_report → reporting/build_site → reporting/evaluate_forecasts
- Utilities (tools/): time_split_validate, backfill_forecast_history, scan_wa_bonds_zip

Run the full pipeline: `python -m src.cli one-click`


### End‑to‑end (one‑off)
```bash
# from repo root
python -m src.cli run-all
python -m src.reporting.validate_and_report
python -m src.reporting.evaluate_forecasts  # writes spreadsheets; scores if actuals exist
```

### Individual steps (iterate)
```bash
python -m src.data_ingest.fetch_bonds
python -m src.data_ingest.process_bonds
python -m src.data_ingest.map_poa_sa2
python -m src.models.nowcast
python -m src.models.forecast
python -m src.reporting.validate_and_report
python -m src.reporting.evaluate_forecasts
```

### Makefile shortcuts (recommended)

Common tasks are wrapped in a Makefile:

```bash
# Fit latest with recency=6, bias-correct, isotonic calibration, prior-shift
make fit_latest

# Evaluate a window (e.g., the last two months)
make eval_recent START=2025-07 END=2025-08

# Rebuild the static site (writes docs/)
make site

# One-click monthly: fit → eval last 2 months → build site
make all_monthly

# Print Average Precision for months (requires sklearn)
make pr_months MONTHS="2025-07 2025-08"

# Serve the site locally at http://localhost:8081 (PORT overridable)
make serve

# Export Top‑20 alerts for a month at a fixed threshold (default latest, THR=0.34)
make alerts_top20 THR=0.30 MONTH=2025-09

# Export Top‑20 alerts at the monthly best threshold (from evaluation summary)
make alerts_best MONTH=2025-08
```

### Operational guidance

- Thresholds: use the per‑month `best_thr_f1` from `outputs/evaluations/forecast_eval_summary.csv` for alerts instead of a fixed 0.50. The site and JSON exports include these alerts.
- Calibration: enable isotonic calibration to tighten Brier/log‑loss. Validation supports `--calibrate-isotonic`, and the latest forecast supports `--calibrate-isotonic` as well.
- Recency: default `recency_half_life=6` (also in `src/one_click.py`) emphasizes recent dynamics and improved late‑month performance.
- Prior‑shift: `src.models.forecast` supports `--prior-shift` to align logits to recent base rates (env `PRIOR_MONTHS`, default 3). Combine with isotonic.

### New evaluation outputs

`src.reporting.evaluate_forecasts` now also writes:
- Expected Calibration Error (`ece`) and Brier decomposition (`brier_reliability`, `brier_resolution`, `brier_uncertainty`) into `forecast_eval_summary.csv`.
- Precision‑Recall curves with Average Precision (PNG per month) into `outputs/figures/`.

### Site improvements

- The site JSON (docs/data/YYYY‑MM.json) includes an alert flag `a` per SA2 based on the monthly best threshold; the map UI supports an “Alerts only” toggle.
- The footer shows AUC, Brier, LogLoss, and the monthly threshold.

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

### Nowcast (`src/models/nowcast.py`)
- **Likelihood**: Negative Binomial over **disposals** (ended tenancies).  
- **Offset**: `log(stock_bonds)` (exposure), so the expected count scales with stock.  
- **Structure**:
  - SA2 random intercepts (hierarchical pooling)
  - Linear time trend
  - Month‑of‑year seasonality (sum‑to‑zero)
  - Overdispersion parameter `alpha_nb`
- **Output**: `availability_rate` = posterior mean of expected count per unit stock.

### Forecast (`src/models/forecast.py`)
- **Target**: `1{ Δ median_rent_{t+1} > g }`, where `g = 2%` (configurable).  
- **Predictors** (standardized):
  - Core: `availability_rate`, `churn_rate`, `rent_mom_1m`, `rent_mom_3m`
  - **WA-demeaned variants**: (`*_wa_dev`) capture how far each SA2 sits from the statewide monthly mean.
  - **Interaction terms**: `availability_rate × rent_mom_1m`, `availability_rate × churn_rate`, `churn_rate × rent_mom_3m` highlight compounding pressure.
  - Calendar, lags, external signals, neighbour availability (when present).
- **Structure**:
  - Logistic regression with **SA2 random intercepts** + mild varying slopes for key drivers.
  - Automatic isotonic calibration (when sufficient realised history exists), optional prior-shift and bias correction.
- **Outputs**:
  - `price_pressure_prob` per SA2 (latest month) with posterior intervals.
  - `outputs/evaluations/forecast_training_metrics.json` summarising training AUC/PR and the recommended operating threshold (max F1 over 0.05–0.95 grid).

> **Extensions** (easy, optional): add supply-side signals (approvals, completions) to `external_signals.parquet`—they automatically propagate through the feature builder.

---

## Evaluation & back‑testing

`src/reporting/evaluate_forecasts.py` automatically:
- Builds **realized labels** when the next month’s SA2 panel is available.
- Scores predictions month‑by‑month: **AUC**, **Brier**, **log‑loss**, precision/recall at 0.5.
- Adds skill/error metrics:
  - **MASE** vs baselines: `mase_vs_seasonal` (t−12), `mase_vs_locf` (t−1)
  - **Brier Skill Score** vs both baselines: `bss_vs_seasonal`, `bss_vs_locf`
  - **WA‑weighted errors** (by `stock_bonds`): `w_mae`, `w_rmse`
- Writes a **calibration plot** (predicted vs observed probability).

Until the next month’s panel exists, it will print:  
**“No realized months yet to score.”** (expected)

---

## Time‑Split & Walk‑Forward

This repo includes a helper to train on an early block of months and validate on later months using a strict split (no leakage), plus an optional walk‑forward variant that refits per target month.

Prerequisites: run up to SA2 staging so `data_stage/bonds_panel_sa2.parquet` covers the period of interest.

```
python -m src.data_ingest.fetch_bonds
python -m src.data_ingest.process_bonds
python -m src.data_ingest.map_poa_sa2
```

Example: Train on Jan–Apr, validate on Jun–Sep

1) Fixed split (single fit):
```
python -m tools.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09
  --draws 400 --tune 400 --chains 1 --cores 1
```
This:
- Fits the Negative Binomial nowcast on Jan–Apr and predicts out‑of‑sample `availability_rate` for the base months (May–Aug).
- Fits the logistic forecast on Jan–Apr and produces probabilities for the target months (Jun–Sep).
- Appends predictions to `data_stage/price_pressure_forecast_sa2_history.parquet`.

2) Walk‑forward (operational realism):
```
python -m tools.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09 \
  --walk-forward \
  --draws 400 --tune 400 --chains 1 --cores 1
```
For each target month T in Jun..Sep, this refits:
- Nowcast on all data up to and including T−1 (to estimate availability for base month T−1);
- Logistic on months from Jan up to T−2 (so labels do not use information at T).

Evaluate just these months (once actuals exist for each):
```
python -m src.reporting.evaluate_forecasts --start 2025-06 --end 2025-09
```
Rebuild the static site (adds/updates docs/data/*.json):
```
python -m src.reporting.build_site
```

Extras (feature switches):
- `--with-external`: merge monthly external signals (from `data_stage/external_signals.parquet`).
- `--extra-features`: add calendar features and simple lags.
- `--with-spatial`: include neighbor‑mean availability `nbr_availability_rate` (see below).
- `--leakage-canary`: permute train labels; metrics should collapse if there’s no leakage.
- `--recency-half-life 12`: up‑weight recent months (half‑life = 12 months) in both fits.

---

## Website (static site + Netlify)

The repo contains a generator that builds a static site into **`docs/`**:

```bash
python -m src.reporting.build_site
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
python -m src.reporting.build_site
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

### Map metrics & legend

- Forecast probability: colors interpolate from light yellow (0) to deep red (1) to show Pr[Δrent > 2%].
- Actual outcome: red = rise > 2%, green = not rise, grey = n/a (no realized data yet).
- Correct @ 0.60: uses a 0.60 cutoff to classify a “predicted raise” (prob ≥ 0.60). Green = prediction matches actual; red = prediction does not match; grey = n/a. This cutoff is visualization‑only.
- Error (prob − actual): blue (negative) → grey (0) → red (positive) diverging scale.
- Abs. error |prob − actual|: light to dark green with higher values darker.

Note: summary metrics in the footer (precision/recall/accuracy) are computed at a 0.50 threshold and are not affected by the “Correct @ 0.60” view.

### Validation options (quick wins)

- `--leakage-canary`: permutes train labels inside each fold; metrics should collapse if there’s no leakage.
- `--extra-features`: adds calendar features (mo_sin/mo_cos, quarter, EOFY, uni semester flags) and simple lags
  (`availability_rate_lag1`, `churn_rate_lag1`, `rent_mom_12m`).
- `--recency-half-life <months>`: up-weights recent months in both the nowcast and forecast fits
  via a half-life schedule (e.g., 12 → weights halve every 12 months).

CLI flags are also available on:

```bash
# Optional recency weighting during one-shot fits
python -m src.models.nowcast --recency-half-life 12
python -m src.models.forecast --recency-half-life 12
```

Spatial feature option (neighbor mean availability):
```bash
python -m tools.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09 \
  --with-spatial
```

Neighbor feature details:
- `nbr_availability_rate` is the mean of neighboring SA2s’ `availability_rate` for the same month (queen adjacency).
- Uses the same‑month availability from the nowcast/prediction for base months, so there’s no label leakage.
- Build adjacency once with `python -m tools.spatial_features` or let the validator auto‑create it on first run.

---

## Automation (cron / WSL)

Use `scripts/monthly_job.sh` to run the full pipeline and publish the site. Example crontab (7:30 on the 5th each month):

```
30 7 5 * * /home/<you>/projects/wa_rental_pressure/scripts/monthly_job.sh
```

The script does:
1) Activate venv  
2) `src.cli.commands.run_all` → `src.reporting.evaluate_forecasts`  
3) `src.reporting.build_site`  
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
- If the ABS correspondence is a **mega‑ZIP mis‑saved as .xlsx**, `data_ingest/map_poa_sa2.py` will detect it and open the inner file.  
- If CSV has weird delimiters/encodings, the loader **sniffs** them.  
- Script prints: `poa=... sa2=... ratio=... scale=ratio/percent rows=...`

**Timestamp not JSON serializable**  
- Fixed: the site builder strips/normalizes datetimes before serializing.

**Folium map feels slow**  
- Geometries are simplified in meters; raise tolerance for more speed/smaller HTML.

**PyMC `MutableData` missing / BLAS warning**  
- Not used: the models work with NumPy arrays (PyMC 5).  
- BLAS warning = performance only; Conda builds usually resolve it.

**PyTensor constant_folding / type errors during sampling**  
- For detailed traces, set: `export PYTENSOR_FLAGS="optimizer=fast_compile,exception_verbosity=high"` and rerun.
- Ensure the compile cache is writable: `export PYTENSOR_FLAGS="${PYTENSOR_FLAGS},base_compiledir=./.pytensor"`.
- Use single-core dev settings while debugging: `--chains 1 --cores 1`.
- The `src.cli.commands.one_click` entrypoint already sets `base_compiledir=./.pytensor` if not present.

**Keep the repo small**  
See `.gitignore` below; do **not** commit `data_*` or anything in `outputs/`.

---

## Roadmap / ideas

- **Features**: `lodge_rate`, `stock_growth_1m`; seasonality in logistic.  
- **Pooling**: SA3 random intercepts above SA2.  
- **Calibration**: Platt/isotonic scaling if over/under‑confident.  
- **Notifications**: Slack/Discord webhook summary after each run.  
- **Distribution**: `environment.yml` for Conda users.

## TODO (next run cycle)

- **Calibration**: implement the rolling logit temperature + intercept calibrator (and isotonic-with-shrink fallback) on out-of-fold predictions; wire it into `validate_and_report.py` / `evaluate_forecasts.py` and select by time-series CV.
- **Thresholds**: adopt a non-0.5 operating rule (default θ≈0.33 with ±0.03 guardrail) and add support for capacity-based top-k thresholds in the evaluation/export path.
- **Temporal features & weights**: add recency/seasonality features (1- & 3-month deltas, month-of-year, time-since-spike) and time-decay sample weights before the next LightGBM fit.
- **LightGBM retune**: rerun with higher capacity/regularisation (num_leaves=63, min_data_in_leaf=120, lambda_l2=10, feature_fraction/bagging_fraction=0.8, bagging_freq=5, rounds≈1200, early_stop≈200) under a rolling-origin CV scoring the last two months.
- **Stacking**: refit the meta-learner on logit(p_pyMC), logit(p_LGBM) with a regularised logistic model, then pass its output through the chosen calibrator.
- **Monitoring**: add monthly reliability/ECE charts, predicted vs observed prevalence plots, and precision/recall + alert counts at the production threshold so calibration drift is obvious.
- Refresh macro parquet via `python -m src.data_ingest.external_signals` (or `bash scripts/fetch_abs_wa.sh`) before the next full pipeline run so the WA ABS feeds stay current.
- Re-run `python -m src.cli one-click -- --recency-half-life 6 --calibrate-isotonic --prior-shift` to benchmark production accuracy with the new calibration + prior shift options enabled once the calibrators are in place.
- Compare the updated metrics JSON against the previous release and capture any lift/bias notes in `docs/CODEMAP.md` or the evaluation notebook.

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

---

## External signals (APIs)

You can augment the forecast with macro signals via APIs, with CSV fallbacks:

- Supported signals (monthly):
  - RBA target cash rate (parsed from RBA page; no key required)
  - WA unemployment rate (ABS Data API SDMX‑JSON via curl helper)
  - WA building approvals (ABS Data API SDMX‑JSON via curl helper)

How it works:
- `python -m src.data_ingest.external_signals` writes `data_stage/external_signals.parquet` with columns
  `month`, `rba_cash_rate`, `wa_unemp_rate_sa`, `wa_build_approvals_num` (with lagged versions auto-added during modelling).
- `scripts/external_signals.env.sample` documents API overrides; copy to `.env` or `scripts/external_signals.env` to customise.
- `scripts/fetch_abs_wa.sh` is a CLI helper that materialises the same ABS series into CSV fallbacks under `data_raw/external/` (requires `curl` + `jq`).
- `tools/time_split_validate.py` consumes the shared feature helpers and accepts `--with-external` so validation runs mirror production signals.

API configuration (optional; ABS curl helper runs with sensible defaults):
- Override ABS dataflow/key if you need a different series:
  - `ABS_UNEMPLOYMENT_DATAFLOW` (default `LF`)
  - `ABS_UNEMPLOYMENT_KEY` (default `M13.3.1599.20.5.M` → SA unemployment rate, Persons, Total age, WA)
  - `ABS_UNEMPLOYMENT_START` (default `2010-01`)
  - `ABS_BUILDAPP_DATAFLOW` (default `BA_GCCSA`)
  - `ABS_BUILDAPP_KEY` (default `1.1.9.TOT.TOT.10.5.M` → Number of dwelling units, Total sectors/work/building, Original, WA)
  - `ABS_BUILDAPP_START` (default `2015-01`)
- You can still supply full SDMX URLs or CSV fallbacks if necessary:
  - `ABS_UNEMPLOYMENT_SDMX_URL`, `ABS_BUILDAPP_SDMX_URL`
  - `UNEMPLOYMENT_CSV_URL`, `BUILDING_APPROVALS_CSV_URL`
  - `RBA_CASHRATE_CSV_URL` (optional override for the cash rate)

If none of the above are set, the loader falls back to local CSVs under `data_raw/external/`:
- `data_raw/external/unemployment_rate_wa.csv` with columns `month,wa_unemp_rate_sa`
- `data_raw/external/building_approvals_wa.csv` with columns `month,wa_build_approvals_num`

Example usage (bash):
```bash
# override ABS defaults only if you need a different series
export ABS_UNEMPLOYMENT_KEY="M13.3.1599.30.5.M"  # trend instead of seasonally adjusted
python -m src.data_ingest.external_signals
python -m tools.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09 \
  --with-external
```

Notes:
- ABS Data API calls are made via `curl` (user-agent friendly); responses are parsed in Python.
- Any missing series are skipped gracefully; the pipeline proceeds with whichever columns are available.

### Optional data quality and bias toggles

- Ingest winsorization (reduce outlier influence in raw lodgements):
  - Set `WINSORIZE_RENT=1` before running `src.data_ingest.process_bonds` (or `src.cli.commands.one_click`).
  - Quantiles configurable via `WINSOR_LO`/`WINSOR_HI` (defaults `0.01/0.99`).

- Forecast per‑SA2 bias correction (light calibration on last K realized months):
  - Add `--bias-correct-l6` to `python -m src.models.forecast` (uses K=6 by default).
  - Or set `BIAS_CORRECT_L6=1` and optionally `BIAS_CORR_MONTHS=K`.
  - `BIAS_CORR_GAMMA` scales the correction strength (default `0.25` for mild correction).
  - This subtracts `gamma × mean_residual` over the last K months from the new prediction and clips to [0,1]. The raw (uncorrected) probability is kept in `price_pressure_prob_raw`.

Quick experiments (spatial + external + calendar + recency):
```bash
# Build signals and adjacency once
python -m src.data_ingest.external_signals
python -m tools.spatial_features

# Fixed split
python -m tools.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09 \
  --with-external --extra-features --with-spatial \
  --recency-half-life 12 --draws 400 --tune 400 --chains 1 --cores 1

python -m src.reporting.evaluate_forecasts --start 2025-06 --end 2025-09

# Walk-forward (refits per target month)
python -m tools.time_split_validate \
  --train-start 2025-01 --train-end 2025-04 \
  --val-start 2025-06 --val-end 2025-09 \
  --walk-forward --with-external --extra-features --with-spatial \
  --recency-half-life 12 --draws 400 --tune 400 --chains 1 --cores 1

python -m src.reporting.evaluate_forecasts --start 2025-06 --end 2025-09
```

### Optional GBM baseline (advanced)

If you want a quick non‑Bayesian baseline for comparison/stacking:

```bash
pip install lightgbm scikit-learn
python -m src.models.nowcast --draws 400 --tune 400 --chains 1 --cores 1
python -m tools.gbm_baseline
```

This writes `outputs/gbm_baseline_latest.parquet` with `p_gbm` for the next month, using the same core features (availability_rate, churn, rent momentum). You can use it to sanity‑check or as one input to a simple stacker.

## New defaults and stronger validation

One-click (`python -m src.cli one-click`) now pins sensible defaults that gave small but consistent gains in calibration and weighted error on recent months:

- Data quality: winsorize lodgement rents at 5–95% (override via `WINSOR_LO/H I`).
- Features: external signals, spatial neighbor mean availability, calendar dummies.
- Recency: half‑life weighting (12 months) in fitting.
- Validation: walk‑forward across the validation window with stacking and calibration.
  - Stacking blends the Bayesian logistic with a boosted‑tree model (LightGBM if available), and isotonic calibration aligns probabilities to observed frequencies.
- Forecast: same enriched features as validation, with mild bias correction (`BIAS_CORR_GAMMA=0.25`, window `BIAS_CORR_MONTHS=6`). Raw probabilities are preserved as `price_pressure_prob_raw`.

### Extended default split

If at least 18 months of data exist, the one‑click default split is:
- Train: last 12 months
- Validate: last 6 months (walk‑forward refits)

Otherwise it falls back to a 4‑month train and 4‑month validation in the latest year.

### New validation flags (tools/time_split_validate.py)

- `--use-gbm`: use a boosted‑tree classifier for the forecast stage (LightGBM or sklearn fallback).
- `--stack-gbm`: blend Bayesian logistic and GBM predictions (simple average).
- `--calibrate-isotonic`: apply isotonic calibration on the train window.

Example (longer window, stacked + calibrated):

```bash
python -m tools.time_split_validate \
  --train-start 2024-07 --train-end 2025-06 \
  --val-start 2025-07 --val-end 2025-12 \
  --with-external --extra-features --with-spatial \
  --recency-half-life 12 --stack-gbm --calibrate-isotonic --walk-forward
python -m src.reporting.evaluate_forecasts --start 2025-07 --end 2025-12
```

### Live forecast parity (src/models/forecast.py)

The live forecast now uses the same enriched features as validation when available:
- Calendar dummies, simple lags (`availability_rate_lag1`, `churn_rate_lag1`, `rent_mom_12m`).
- Neighbor mean availability (`sa2_neighbors.parquet`).
- External signals and their lags (`rba_cash_rate`, `wa_unemp_rate_sa`, `wa_build_approvals_num`, plus `_lag1/_lag3`).
- Mild bias correction controlled via `BIAS_CORR_GAMMA` (default `0.25`) and `BIAS_CORR_MONTHS` (default `6`).

### Dependencies

Two optional ML deps were added for stacking/calibration:

- `lightgbm>=4.4` (preferred) and `scikit-learn>=1.5` (fallback + isotonic).

They are included in `requirements.txt`. If LightGBM is unavailable, the code falls back to sklearn’s GradientBoostingClassifier.
