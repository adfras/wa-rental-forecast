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
