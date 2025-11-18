# TODO

- Investigate chronic underpredictions (e.g. Roebourne, Gnowangerup, Morawa, City Beach) and add features that capture rapid rent shocks in low-stock regions.
- Prototype alternative recency weighting or change-point detection to react faster to regime shifts without hurting discrimination.
- Refit September 2025 with the supply-shock + lodgement-weighted configuration so the published site/export JSON reflects the improved calibration.
- Surface tiered alert CSVs on the Netlify site and wire them into the alerting workflow.
- Automate residual dashboards (histograms, residual maps) fed by `outputs/tables/merged_predictions_full_YYYY-MM.csv`.
- Prototype approaches to help thin-data SA2s (e.g., region-level pooling, external signals, booster blend) so sparse series stop getting shrunk to the statewide mean.
- Explore adding a sparsity-aware regional random effect (e.g., super-SA2 clusters) to lift thin-data areas without overfitting.
