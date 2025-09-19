Code Map — WA Rental Forecast

Purpose: clarify what's core vs. optional and show the flow between modules.

Core pipeline (minimal set)
- src/config.py — paths and constants
- src/data_ingest/process_bonds.py — ingest ZIP → postcode×month panel
- src/data_ingest/map_poa_sa2.py — POA→SA2 mapping (filters to WA)
- src/models/nowcast.py — availability nowcast (Negative Binomial)
- src/models/forecast.py — price-pressure forecast (hierarchical logistic)
  - Recency weighting, optional isotonic calibration (`--calibrate-isotonic`),
    optional prior-shift (`--prior-shift`), bias correction on last 6 months,
    and mild varying slopes on key drivers (availability_rate, churn_rate).
- src/reporting/validate_and_report.py — Top-20 CSV, folium map, PNG
- src/reporting/build_site.py — static site into docs/
- src/reporting/evaluate_forecasts.py — scoring (AUC, Brier, log-loss, MASE, Brier Skill, WA-weighted), spreadsheets
- src/data_ingest/fetch_bonds.py — downloads latest AHDAP ZIP (optional if you place files manually)
- src/data_ingest/external_signals.py — optional macro signals (RBA cash rate, WA unemployment, approvals)

Orchestrators
- src/cli/commands/one_click.py — preferred end-to-end driver (train/validate, evaluate, forecast latest, reports, site)
- src/cli/commands/run_all.py — basic pipeline (kept for compatibility); prefer the CLI `one-click`

Utilities (moved to tools/)
- tools/time_split_validate.py — strict time-split + walk-forward validation helper
- tools/backfill_forecast_history.py — merges docs/data/*.json into parquet history
- tools/scan_wa_bonds_zip.py — diagnostic: scan raw ZIP for headers/month coverage
- tools/spatial_features.py — build SA2 adjacency (queen); neighbor features
- tools/print_ap.py — print Average Precision (PR AUC) for specified months
- tools/export_alerts.py — export Top‑20 alerts at a fixed threshold for a month
- tools/export_alerts_best.py — export Top‑20 alerts using monthly best_thr_f1 from eval summary

Dependency sketch

    AHDAP ZIP ─┐                              ABS assets ─┐
               ├─ process_bonds  (postcode)                │
               │                                          │
               └──────────────► map_poa_sa2 ───────────────┘
                                     │
                                     ▼
                           model_nowcast  → availability_nowcast_sa2.parquet
                                     │
                                     ▼
                           model_forecast → price_pressure_forecast_sa2.parquet (+history)
                                     │
                                     ├─ validate_and_report (Top-20, map, PNG)
                                     ├─ evaluate_forecasts (scores once actuals exist)
                                     └─ build_site (docs/ index + data)

Notes
- Day-to-day, use `python -m src.cli one-click` or the core step-by-step list.
- Utilities live in `tools/` to declutter `src/`.
- Validation flags to know: `--with-external`, `--with-spatial`, `--extra-features`, `--walk-forward`, `--recency-half-life`, `--leakage-canary`, `--calibrate-isotonic`.
- Latest forecast options: `--recency-half-life`, `--bias-correct-l6`, `--calibrate-isotonic`, `--prior-shift`.
- One‑click defaults now use `recency_half_life=6`.
