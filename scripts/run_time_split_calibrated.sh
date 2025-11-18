#!/usr/bin/env bash

# Heavy walk-forward validation with raw-prob isotonic calibration.
# Usage: bash scripts/run_time_split_calibrated.sh [extra args passed to time_split_validate]

set -euo pipefail

: "${CALIB_USE_RAW:=1}"
export CALIB_USE_RAW

python -m tools.time_split_validate \
  --train-start 2023-03 \
  --train-end 2025-02 \
  --val-start 2025-03 \
  --val-end 2025-08 \
  --walk-forward \
  --recency-half-life 6 \
  --with-external \
  --extra-features \
  --with-spatial \
  --draws "${DRAWS:-1800}" \
  --tune "${TUNE:-1800}" \
  --chains "${CHAINS:-6}" \
  --cores "${CORES:-6}" \
  --pymc-target-accept "${TARGET_ACCEPT:-0.995}" \
  --calibrate-isotonic \
  "$@"
