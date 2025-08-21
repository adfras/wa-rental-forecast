#!/usr/bin/env bash
set -euo pipefail

# CHANGE these paths if your project lives elsewhere
PROJ="/home/alan/projects/wa_rental_pressure"
VENV="$PROJ/.venv"

cd "$PROJ"
source "$VENV/bin/activate"

timestamp=$(date +'%Y%m%d_%H%M')
mkdir -p logs

# Run the full pipeline (fetch → process → map → nowcast → forecast → report)
python -m src.run_all | tee "logs/run_all_${timestamp}.log"

# Score predictions against realized data (if available) + export spreadsheet
python -m src.evaluate_forecasts | tee -a "logs/run_all_${timestamp}.log"

echo "Done at $(date)."
