#!/usr/bin/env bash
set -euo pipefail

# Make cron-friendly: ensure common bin dirs are on PATH (cron often lacks /usr/local/bin)
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# CHANGE these paths if your project lives elsewhere
PROJ="/home/alan/projects/wa_rental_pressure"
VENV="$PROJ/.venv"

cd "$PROJ"
if [ ! -d "$VENV" ]; then
  echo "[ERROR] venv not found at $VENV. Create it and install requirements." >&2
  exit 1
fi
source "$VENV/bin/activate"

timestamp=$(date +'%Y%m%d_%H%M')
mkdir -p outputs/logs

# Run the full pipeline (fetch → process → map → nowcast → forecast → report)
{
  echo ">>> $(date) python -m src.cli run-all"
  python -m src.cli run-all

  echo ">>> $(date) python -m src.reporting.evaluate_forecasts"
  # Ensure history includes all months from site JSON before evaluating
  python -m tools.backfill_forecast_history || true
  python -m src.reporting.evaluate_forecasts || true

  echo ">>> $(date) python -m src.reporting.build_site"
  python -m src.reporting.build_site

  echo ">>> $(date) which netlify"
  command -v netlify || echo "netlify CLI not found (skipping deploy)"

  if command -v netlify >/dev/null 2>&1; then
    echo ">>> $(date) netlify deploy --dir=docs --prod"
    # Publish to Netlify using the linked project in .netlify/state.json
    netlify deploy --dir="$PROJ/docs" --prod
  fi

  echo "Done at $(date)."
} | tee "outputs/logs/run_all_${timestamp}.log"
