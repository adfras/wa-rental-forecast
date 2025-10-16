# WA Rental Forecast

**Bayesian nowcast + forecast of rental price pressure at SA2 level (Western Australia).** Outputs a Top‑20 list, an interactive map, and a spreadsheet of probabilities that **next month’s median rent rises by more than 2 %** (default threshold), with automatic monthly back‑testing once new data lands.

- **Live site:** https://wa-rental-forecast.netlify.app
- **Last end-to-end update:** 16 October 2025 (includes lodgement-aware smoothing, sparse-market features, and booster diagnostics.)

---

## 1. Documentation Index & Project Map

| Topic | Location | Notes |
|-------|----------|-------|
| Code flow & responsibilities | [docs/CODEMAP.md](docs/CODEMAP.md) | Core vs optional modules, orchestration entrypoints. |
| Data dictionary | [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Column-level descriptions for staged parquet + site JSON. |
| Experiment history | [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | Chronological run settings, sampler configs, key metrics.<br/>Entry **ID 9 (2025‑10‑16)** logs the sparse-market booster & diagnostics. |
| Lodgement weighting rationale | [docs/reports/2025-10-15_lodgement_weighting.md](docs/reports/2025-10-15_lodgement_weighting.md) | Design note for supply-shock + lodgement weighting refit. |
| Outstanding work | [TODO.md](TODO.md) | Research backlog; focus areas called out in sparse SA2 mitigation. |
| Site JSON snapshots | `docs/data/` | One file per month per threshold; consumed by Netlify site. |

For a guided walkthrough of the codebase, start with [docs/CODEMAP.md](docs/CODEMAP.md) and follow the links above.

---

## 2. Operational Runbook (Monthly Release)

### 2.1 Pre-flight checklist

1. **Environment** – Activate the venv (`source .venv/bin/activate`). Verify `make --version` and `python --version` (>=3.12).  
2. **Data freshness** – Run `python -m src.data_ingest.fetch_bonds` if the AHDAP CKAN has a new ZIP; confirm `data_stage/bonds_panel_sa2.parquet` ends at last month.  
3. **Diagnostics (sparse SA2s)** – `python tools/sparse_sa2_diagnostic.py` generates `outputs/tables/sparse_sa2_summary.csv` and plots in `outputs/figures/`. Review Roebourne, Gnowangerup, Morawa, City Beach before rerunning the forecast.

### 2.2 Standard monthly run (heavy configuration)

```bash
source .venv/bin/activate
make fit_latest          # walk-forward 2023-03…latest-1 with 6×1 800 draws/tune, isotonic calibration
make eval_recent START=2025-08 END=$(date -d "$(date +%Y-%m-01) -1 month" +%Y-%m)
make site                # rebuild docs/ for Netlify
```

This resets `data_stage/price_pressure_forecast_sa2_history.parquet` and rewrites multi-threshold histories (`thr0p010`, `thr0p020`, `thr0p030`) using the new sparse-market features and booster logit.

### 2.3 Post-run publishing

1. **Calibration exports** – `python -m scripts.evaluate_calibration --month <latest>` populates tiered alert CSV/JSON in `outputs/tables/`.  
2. **Static site deploy** – `netlify deploy --dir=docs --prod` (or run `scripts/monthly_job.sh`).  
3. **Log the run** – Append sampler diagnostics & highlights in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) if metrics change and capture sparse-SA2 bias shifts.

### 2.4 Day-zero triage (before data drop)

- Refresh external signals (`python -m src.data_ingest.external_signals`).  
- Inspect `outputs/tables/worst_underpredictions_*.csv` for any SA2 outside the sparse cohort needing ad-hoc investigation.  
- Queue feature experiments via `tools/time_split_validate.py` with custom flags (see §4).

---

## 3. Default Configuration Snapshot (as of October 2025)

| Setting | Default | Where | Notes |
|---------|---------|-------|-------|
| Training window | 2023-03 → month before validation start | `Makefile` | `VAL_END` auto-resolves to previous month; `VAL_START` is rolling 6 months; `TRAIN_END` backs off one month. |
| Chains × cores | `CHAINS=6`, `CORES=6` | `Makefile`, `scripts/run_time_split_calibrated.sh` | Saturates typical 12-thread boxes; adjust per hardware. |
| Draws / Tune | `DRAWS=1800`, `TUNE=1800` | Same | Aligns with ID8/ID9 experiments. |
| Target accept | `TARGET_ACCEPT=0.995` | Same | Stabilises heavy tails post lodgement-smoothing. |
| Calibration | `--calibrate-isotonic`, `CALIB_USE_RAW=1` | Makefile & scripts | Raw-prob isotonic, applied both OOS validation and latest forecast. |
| Feature toggles | `--with-external --extra-features --with-spatial --walk-forward` | Makefile | Brings in macro signals, WA aggregates, adjacency averages. |
| Sparse-market extras | Enabled by default | `src/models/forecast.py` | Lodgement scarcity ratios, rolling spike magnitudes, thin-market interactions, booster logit. |
| Booster | On (`SPARSE_MARKET_BOOSTER=1`) | Forecast build | GradientBoostingClassifier (300 trees, depth≤3) feeding `booster_logit`; disable with env var for A/B comparisons. |
| Lodgement weighting | `LODGE_SMOOTHING=12`, `LODGE_WEIGHT_FLOOR=0.05` | Forecast env | Down-weights months with few lodgements. |

Override any of these on the command line, e.g. `make fit_latest DRAWS=1200 CORES=8`.

---

## 4. Diagnostics, Experiments & Sparse-SA2 Toolkit

- **Sparse SA2 dashboard** – `tools/sparse_sa2_diagnostic.py` (see §2.1) summarises bias, lodgement coverage, and rent shocks; outputs land in `outputs/tables/sparse_sa2_*`. Use before/after refits.
- **Time-split validator** – `tools/time_split_validate.py` is the canonical way to compare configurations. Key flags extracted from production runs:  
  `--sampler-rhat-max 1.01 --sampler-retries 2 --retry-draw-multiplier 1.5` keep R-hat in check; `--threshold-grid 0.01 0.02 0.03` writes multi-threshold histories for evaluation.
- **Experiment log** – Update [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) after each heavyweight run. Entry ID9 captures the sparse-market booster; use the same format to add future experiments.
- **Report archive** – `docs/reports/2025-10-15_lodgement_weighting.md` documents the supply-shock + lodgement weighting changes rolled into the current defaults. Add new reports here for major refactors.

---

## 5. Quick Start & Environment Setup

### Option A — Python virtualenv (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Option B — Conda/Mamba (for optimised BLAS)
```bash
mamba create -n wa-rental-forecast python=3.12 pymc arviz pytensor pandas numpy \
  pyarrow geopandas pyogrio shapely folium branca matplotlib requests tqdm openpyxl
mamba activate wa-rental-forecast
```

> If PyTensor warns about BLAS, Conda/Mamba builds usually resolve it automatically.

Activate the chosen environment before running any `make` targets or scripts.

---

## 6. Pipeline Overview & Repository Layout

### Project goals
- **Nowcast**: estimate current **rental availability** (tightness) for each SA2 in WA.  
- **Forecast**: probability that **next month’s median rent** increases by **> 2 %** (configurable).

Monthly deliverables:
- **Top‑20 risers CSV**
- **Interactive SA2 map** (folium HTML)
- **Static bar chart PNG**
- **Spreadsheet of all SA2 probabilities** (with SA2 names)
- **Evaluation files** after the next month’s data arrives (AUC/Brier/Calibration)

### Pipeline overview
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
           models/forecast.py (hier. logistic + sparse-market booster)
                         ↓
   data_stage/price_pressure_forecast_sa2.parquet
                         ↓
      reporting/validate_and_report.py + reporting/evaluate_forecasts.py
      ├─ outputs/tables/top20_pressure_risers.csv
      ├─ outputs/tables/price_pressure_forecast_sa2_latest[_named].xlsx
      ├─ outputs/reports/map_price_pressure.html
      ├─ outputs/figures/top20_pressure_risers.png
      ├─ outputs/evaluations/forecast_eval_summary.csv (once actuals exist)
      └─ outputs/figures/forecast_calibration_YYYY-MM.png (once actuals exist)
```

### Repository layout
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

tools/
  time_split_validate.py      # Walk-forward validator (heavy runs)
  sparse_sa2_diagnostic.py    # New thin-market bias diagnostics
  backfill_forecast_history.py, export_* utilities, etc.

docs/
  data/                       # Static site payloads (monthly JSON, thresholds)
  reports/                    # Deep-dive write-ups (lodgement weighting, …)
  CODEMAP.md, DATA_DICTIONARY.md, EXPERIMENT_LOG.md

data_raw/                     # (ignored) AHDAP ZIPs, ABS GeoPackage, correspondences
data_stage/                   # (ignored) parquet intermediates
outputs/                      # (ignored) generated artifacts (tables/figures/reports/...)
```

> **Keep the repo light.** `.gitignore` excludes all `data_*` directories and `outputs/`; never commit raw data or generated artifacts.

---

## 7. Data Sources

- **WA Rental Bonds** (AHDAP CKAN): monthly lodgements, disposals, bonds held.  
- **ABS POA→SA2 correspondence (2021)**: population-weighted allocation ratios.  
- **ABS ASGS 2021 SA2 GeoPackage**: geometries & names for SA2s.

Place these in `data_raw/`:
```
data_raw/
  <AHDAP_SLUGS.../ZIPs...>
  ASGS_2021_SA2.gpkg
  CG_POA_2021_SA2_2021.xlsx   # or the ABS mega-ZIP containing it
```
`data_ingest/map_poa_sa2.py` handles CSV/Excel/mega-ZIP edge cases.

---

## 8. Commands & Shortcuts

### CLI & orchestration
- `python -m src.cli one-click` — preferred end-to-end pipeline.
- `python -m src.cli run-all` — compatibility runner; see [docs/CODEMAP.md](docs/CODEMAP.md) for flow.

### Makefile (after activating venv)
```bash
make fit_latest                                # heavy walk-forward + calibration (defaults described in §3)
make eval_recent START=2025-08 END=2025-09     # score a custom window
make site                                       # regenerate docs/
make all_monthly                                # fit → eval (last 2 months) → site → tiered alerts
make pr_months MONTHS="2025-07 2025-08"    # Average Precision per month
make serve PORT=8081                            # serve docs/ locally
make alerts_top20 THR=0.30 MONTH=2025-09        # export fixed-threshold alert list
make alerts_best MONTH=2025-09                  # export best-threshold alert list
```

### Scripts & utilities
- `python -m tools.time_split_validate --help` — full list of validation flags.  
- `python -m tools.export_alerts` / `tools.export_alerts_best` — CSV exports for alerting workflows.  
- `python -m scripts.evaluate_calibration --month YYYY-MM` — writes calibrated probabilities and tiered alerts.  
- `python -m tools.model_experiments` — sandbox for logistic vs gradient-boost experiments.  
- `python tools/sparse_sa2_diagnostic.py` — sparse SA2 bias deep dive (plots + tables).

---

## 9. Outputs & Interpretation

Key artifacts (all ignored by git):
- `outputs/tables/top20_pressure_risers.csv` — Top‑20 SA2s by probability.  
- `outputs/tables/price_pressure_forecast_sa2_latest[_named].xlsx` — Latest probabilities (code-only and name-labelled).  
- `outputs/reports/map_price_pressure.html` — Interactive folium map.  
- `outputs/figures/top20_pressure_risers.png` — Static bar chart.  
- `outputs/evaluations/forecast_eval_summary.csv` — Month-level metrics (AUC, Brier, log-loss, precision/recall).  
- `outputs/evaluations/forecast_eval_details_YYYY-MM.csv` — Row-level truth vs probability.  
- `outputs/figures/forecast_calibration_YYYY-MM.png` — Calibration curve.  
- `outputs/tables/alerts_tiers_YYYY-MM*.csv/json` — Calibrated alert lists (critical/high/medium).  
- `outputs/tables/sparse_sa2_summary.csv`, `outputs/tables/sparse_sa2_timeseries.csv` — Thin-market diagnostics.  
- `outputs/figures/sparse_sa2_*.png` — Per-SA2 diagnostic charts.

**Interpretation**: a forecast value of **0.82** means an 82 % chance that next month’s median rent in that SA2 rises by more than 2 %.

---

## 10. Model Architecture Highlights

### Nowcast (`src/models/nowcast.py`)
- Negative Binomial on disposals with `log(stock_bonds)` exposure.  
- SA2 random intercepts, linear trend, seasonality (sum-to-zero), overdispersion `alpha_nb`.  
- Outputs posterior mean `availability_rate`.

### Forecast (`src/models/forecast.py`)
- Logistic model on `1{Δ median_rent_{t+1} > g}`, `g=0.02` by default.  
- Feature set includes availability/churn/momentum, WA-demeaned variants, interactions, calendar, lags, external signals, neighbour availability, **plus sparse-market features** (lodgement scarcity ratios, recent spike magnitudes, thin-market indicators) and a **GradientBoostingClassifier booster logit**.  
- Hierarchical structure: SA2 random intercepts, mild varying slopes (availability, churn, price band, rent momentum), month-level random effects, price-cluster effect.  
- Lodgement-aware smoothing shrinks volatility features when coverage is thin; lodgement weights also influence pseudo-observations.  
- Optional prior-shift and bias correction; isotonic calibration applied when history exists.  
- Writes posterior summaries and `outputs/evaluations/forecast_training_metrics.json` (threshold sweep, calibration diagnostics).

Detailed feature descriptions live in [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md); architectural context is in [docs/CODEMAP.md](docs/CODEMAP.md).

---

## 11. Evaluation & Back-testing

`src.reporting.evaluate_forecasts` builds realised labels once next-month data is available and scores each month (AUC, Brier, log-loss, precision/recall). It also produces:  
- Expected Calibration Error (ECE) and Brier decomposition fields (`brier_reliability`, `brier_resolution`, `brier_uncertainty`).  
- Precision–Recall curves stored under `outputs/figures/`.  
- Threshold recommendations (`best_thr_f1`) consumed by alert exports and site JSON.

For multi-threshold analysis, run `tools.time_split_validate.py` with `--threshold-grid` and rescore each history using `src.reporting.evaluate_forecasts --threshold <g>`.

---

## 12. Automation & Hosting

- **Netlify deploy**: `scripts/monthly_job.sh` automates venv activation → pipeline → evaluation → site build → optional Netlify CLI deploy. Example cron entry (07:30 on the 5th):  
  `30 7 5 * * /home/<you>/projects/wa_rental_pressure/scripts/monthly_job.sh`
- **Logs**: `outputs/logs/run_all_<timestamp>.log` (ensure cron PATH includes the Netlify CLI).  
- **Credentials**: Run `netlify login` interactively once or set `NETLIFY_AUTH_TOKEN` for headless environments.

---

## 13. Troubleshooting & Performance Tips

- **PyMC / PyTensor**: For verbose diagnostics, `export PYTENSOR_FLAGS="optimizer=fast_compile,exception_verbosity=high"`. Set `PYTENSOR_FLAGS="${PYTENSOR_FLAGS},base_compiledir=./.pytensor"` if cache permissions trip up.
- **Sampler instability**: Lower draws/tune or raise `target_accept` if divergences persist; `tools/time_split_validate.py` already retries with larger draw counts when `rhat > 1.01`.
- **Availability cache**: The validator reuses `data_stage/availability_nowcast_sa2.parquet`. Delete it to force a refit with new features.
- **Geometry issues**: `src.reporting.build_site` simplifies GeoJSON in meters; adjust tolerance inside the script if maps load slowly.  
- **Map metrics stuck at 0.500 cutoff**: the site builder now recomputes per-month best-F1 thresholds from the joined predictions/actuals whenever the evaluation summary is missing a month, so a full `make site` rebuild will refresh the JSON with the correct cutoff and metrics once actuals arrive.
- **Crosswalk quirks**: `data_ingest/map_poa_sa2.py` opens mega-ZIPs masquerading as `.xlsx`; inspect logs for `poa=… sa2=… ratio=…` if mapping looks off.

More tips and open ideas reside in the **Performance tips** and **Roadmap** sections of [docs/CODEMAP.md](docs/CODEMAP.md) and [TODO.md](TODO.md).

---

## 14. Next Steps & Research Backlog

Immediate items from [TODO.md](TODO.md) (keep this synced when priorities change):
- Investigate chronic underpredictions (Roebourne, Gnowangerup, Morawa, City Beach); augment sparse-market features with additional shock signals.  
- Prototype adaptive recency weighting or change-point detection to react faster to regime shifts.  
- Refit September 2025 with supply-shock + lodgement weighting so site/export JSON reflect improved calibration.  
- Surface tiered alert CSVs on the Netlify site and wire into alerting workflow.  
- Automate residual dashboards fed by `outputs/tables/merged_predictions_full_YYYY-MM.csv`.  
- Explore region-level pooling / booster blends for thin-data SA2s; evaluate sparsity-aware random effects.

Keep this section current—if a task moves to "done" or a new one emerges, update both [TODO.md](TODO.md) and the experiment log entry referencing it.

---

Need help? Ping the experiment history ([docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)) for precedent configurations before changing defaults, and log every production run there to keep this README accurate.
