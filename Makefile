PY ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif command -v python >/dev/null 2>&1; then command -v python; \
	else command -v python3; fi)
RECENCY ?= 6
CHAINS ?= 6
CORES ?= 6
DRAWS ?= 1800
TUNE ?= 1800
TARGET_ACCEPT ?= 0.995
PYTENSOR_FLAGS_VALUE ?= blas__ldflags=-L/usr/lib/x86_64-linux-gnu -lopenblas
STAGE_DIR ?= data_stage
TRAIN_START ?= 2023-03

# Dynamically follow the freshest completed month (previous calendar month).
MONTH_ANCHOR ?= $(shell date +%Y-%m-01)
VAL_END ?= $(shell date -d "$(MONTH_ANCHOR) -1 month" +%Y-%m)
VAL_START ?= $(shell date -d "$(VAL_END)-01 -5 month" +%Y-%m)
TRAIN_END ?= $(shell date -d "$(VAL_START)-01 -1 month" +%Y-%m)

PORT ?= 8081

.PHONY: fit_latest eval_recent site all_monthly pr_months serve alerts_top20 alerts_best

# Fit latest forecast with calibration + prior-shift + bias correction
fit_latest:
	rm -f $(STAGE_DIR)/price_pressure_forecast_sa2_history.parquet
	PYTENSOR_FLAGS="$(PYTENSOR_FLAGS_VALUE)" $(PY) -m tools.time_split_validate \
		--train-start $(TRAIN_START) --train-end $(TRAIN_END) \
		--val-start $(VAL_START) --val-end $(VAL_END) \
		--walk-forward --recency-half-life $(RECENCY) \
		--with-external --extra-features --with-spatial \
		--draws $(DRAWS) --tune $(TUNE) --chains $(CHAINS) --cores $(CORES) \
		--pymc-target-accept $(TARGET_ACCEPT) --calibrate-isotonic

# Evaluate a window; pass START=YYYY-MM END=YYYY-MM
eval_recent:
	@if [ -z "$(START)" ]; then echo "Set START=YYYY-MM END=YYYY-MM"; exit 1; fi
	$(PY) -m src.reporting.evaluate_forecasts --start $(START) --end $(END)

# Build the static site (docs/)
site:
	$(PY) -m src.reporting.build_site

# One-shot monthly: fit latest, evaluate last two months, rebuild site
all_monthly: fit_latest
	$(PY) -m src.reporting.evaluate_forecasts --start $$(date -d "$$(date +%Y-%m-01) -2 month" +%Y-%m) --end $$(date -d "$$(date +%Y-%m-01) -1 month" +%Y-%m)
	MONTH_LATEST=$$(date -d "$$(date +%Y-%m-01) -1 month" +%Y-%m); \
	$(PY) -m scripts.evaluate_calibration --month $$MONTH_LATEST --out outputs/tables/alerts_tiers_$${MONTH_LATEST}.json
	$(PY) -m src.reporting.build_site

# Print AP (Average Precision) for a list of months: make pr_months MONTHS="2025-07 2025-08"
pr_months:
	@if [ -z "$(MONTHS)" ]; then echo "Set MONTHS=\"YYYY-MM YYYY-MM ...\""; exit 1; fi
	$(PY) -m tools.print_ap --months $(MONTHS)

# Serve docs/ locally at http://localhost:$(PORT)
serve:
	$(PY) -m http.server --directory docs $(PORT)

# Export Top-20 alerts for a month at threshold THR (defaults: latest month, THR=0.33)
# Usage: make alerts_top20 THR=0.30 MONTH=2025-09
alerts_top20:
	$(PY) -m tools.export_alerts --thr $${THR:-0.33} $${MONTH:+--month $${MONTH}}

# Export Top-20 alerts using per-month best_thr_f1 from evaluation summary (latest by default)
alerts_best:
	$(PY) -m tools.export_alerts_best $${MONTH:+--month $${MONTH}}
