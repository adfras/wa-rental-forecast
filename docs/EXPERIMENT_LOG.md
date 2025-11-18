# Forecast Experiment Log

Tracking major configuration changes, metrics (Mar–Aug 2025 walk-forward unless noted), and diagnostics.

| ID | Date (UTC) | Configuration Summary | Precision@10 | Lift@10 | Precision@20 | Lift@20 | Precision@50 | Lift@50 | AUC | Brier |
|----|------------|-----------------------|--------------|--------|--------------|--------|---------------|---------|-----|-------|
| 1 | 2025-09-22 | Baseline single-pass (old pipeline) | 0.600 | 1.79× | 0.583 | 1.64× | 0.570 | 1.63× | 0.631 | 0.351 |
| 2 | 2025-09-22 | Walk-forward, recency=6, external/calendar/spatial features (auto-cal, tight priors) | 0.750 | 2.29× | 0.750 | 2.23× | 0.680 | 1.91× | 0.710 | 0.210 |
| 3 | 2025-09-22 | + relaxed priors, monthly random effect (current pipeline) | 0.800 | 2.35× | 0.758 | 2.22× | 0.663 | 1.87× | 0.713 | 0.213 |
| 4 | 2025-09-22 | Same as ID3, 400/400 draws (diagnostic) | 0.817 | 2.36× | 0.758 | 2.22× | 0.663 | 1.87× | 0.712 | 0.214 |
| 5 | 2025-09-22 | + price-cluster random effects (walk-forward) | 0.800 | 2.35× | 0.758 | 2.22× | 0.663 | 1.87× | 0.713 | 0.214 |
| 6 | 2025-09-22 | Blended with seasonal naive (w=0.7) | 0.800 | 2.32× | 0.750 | 2.13× | 0.580 | 1.61× | 0.649 | 0.228 |
| 7 | 2025-09-22 | Recency=9 + isotonic calibration | 0.817 | 2.38× | 0.758 | 2.22× | 0.677 | 1.91× | 0.702 | 0.215 |
| 8 | 2025-10-15 | Supply-shock features + raw isotonic + lodgement weighting (6×1 800) | 0.833 | 2.43× | 0.758 | 2.22× | 0.683 | 1.92× | 0.704 | 0.202 |
| 9 | 2025-10-16 | + Sparse-market feature pack (lodgement scarcity, shock recency) + booster logit + sparse diagnostics | – | – | – | – | – | – | pending | pending |

Notes: ID5 (price clusters) produced virtually the same metrics as ID3 (no uplift). ID6 (seasonal-naive blend) degraded ranking and MAE (mase_vs_seasonal ≈ 1.02). ID7 (recency=9 with isotonic) nudged precision@10 up slightly but hurt AUC/Brier, so baseline configuration remains ID3.

Notes:
- ID2 derives from `forecast_eval_summary_rec6_ext_spatial_ch4.csv`. ID3/4 from latest runs (`forecast_eval_summary.csv`, `forecast_eval_summary_lowdraws.csv`).
- Posterior diag for ID3: `max r_hat ≈ 1.005`, `sigma_month ≈ 0.69`, `sigma_a ≈ 0.058`.
- Sept 2025: multi-threshold walk-forward fits now run with `--sampler-rhat-max 1.01`, `--sampler-retries 2`, `--retry-draw-multiplier 1.5` (mirrors production forecast guard) and default to 1 200/1 200 draws when comparing 1 %, 2 %, 3 % thresholds.
- ID9 introduces the sparse-market enhancements (`src/features/engineering.add_sparse_market_features`, `booster_logit`) and the diagnostic script `tools/sparse_sa2_diagnostic.py`; metrics will be logged after the first calibrated release including September 2025 actuals.

Upcoming experiments: spatial segmentation of random effects, seasonal-naïve stacking, alternate walk-forward schedules / calibration tweaks.
