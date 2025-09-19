PY ?= python
RECENCY ?= 6
CHAINS ?= 4
CORES ?= 4
PORT ?= 8081

.PHONY: fit_latest eval_recent site all_monthly pr_months serve alerts_top20 alerts_best

# Fit latest forecast with calibration + prior-shift + bias correction
fit_latest:
	$(PY) -m src.models.forecast \
		--recency-half-life $(RECENCY) \
		--chains $(CHAINS) --cores $(CORES) \
		--bias-correct-l6 --calibrate-isotonic --prior-shift

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
	$(PY) -m src.reporting.build_site

# Print AP (Average Precision) for a list of months: make pr_months MONTHS="2025-07 2025-08"
pr_months:
	@if [ -z "$(MONTHS)" ]; then echo "Set MONTHS=\"YYYY-MM YYYY-MM ...\""; exit 1; fi
	$(PY) -m tools.print_ap --months $(MONTHS)

# Serve docs/ locally at http://localhost:$(PORT)
serve:
	$(PY) -m http.server --directory docs $(PORT)

# Export Top-20 alerts for a month at threshold THR (defaults: latest month, THR=0.34)
# Usage: make alerts_top20 THR=0.30 MONTH=2025-09
alerts_top20:
	$(PY) -m tools.export_alerts --thr $${THR:-0.34} $${MONTH:+--month $${MONTH}}

# Export Top-20 alerts using per-month best_thr_f1 from evaluation summary (latest by default)
alerts_best:
	$(PY) -m tools.export_alerts_best $${MONTH:+--month $${MONTH}}
