# Data dictionary (selected fields)

## Stage files

### `data_stage/bonds_panel_postcode.parquet`
- `postcode` – 4-digit postcode (string)
- `month` – month start (pandas Timestamp)
- `median_rent` – median weekly rent from lodgements
- `p90_rent` – 90th percentile weekly rent
- `count_lodgements` – number of lodgements (new tenancies)
- `mean_days_held` – mean days a tenancy was held (from disposals)
- `count_disposals` – number of disposals/refunds (ended tenancies)
- `stock_bonds` – stock of active rental bonds
- `churn_rate` – (lodgements + disposals) / stock
- `net_stock_change` – lodgements - disposals
- `rent_momentum` – pct change in median rent vs previous month

### `data_stage/bonds_panel_sa2.parquet`
- `sa2_code` – SA2 code (string)
- `month` – month start
- Aggregated counterparts of the postcode panel, weighted via POA→SA2 allocation ratios.

### `data_stage/availability_nowcast_sa2.parquet`
- `sa2_code`, `month`, `availability_rate` – posterior mean of availability per stock.

### `data_stage/price_pressure_forecast_sa2.parquet`
- `sa2_code`, `month`, `price_pressure_prob` – posterior mean probability that next month’s median rent growth exceeds the configured threshold.

## Site JSON (docs/data)

These files power the interactive map in `docs/index.html`:

- `docs/data/months.json` – array of available forecast months as `YYYY-MM`.
- `docs/data/YYYY-MM.json` – object keyed by `sa2_code` with:
  - `p` – forecast probability (Pr[Δrent > threshold]).
  - `l`, `u` – lower/upper bounds of the 90% credible interval (≈ p05, p95).
  - `y` – realized outcome for that month (1 if Δrent > threshold, else 0; may be null if not yet realized).
- `docs/data/summary.json` – per‑month evaluation metrics where realizations exist:
  - `n_sa2`, `base_rate`, `auc`, `brier`, `log_loss`,
  - `precision_at_0_5`, `recall_at_0_5`, `accuracy_at_0_5` (computed at a 0.50 threshold; UI‑only views may use different cutoffs).

UI note: the “Correct @ 0.60” map view classifies SA2s using a 0.60 cutoff (prob ≥ 0.60 ⇒ predict “raise”), color‑coding green when the prediction matches the realized outcome, red when it does not, and grey when the outcome is not yet available. This cutoff is for visualization only and does not affect model training or the 0.50‑based summary metrics above.
