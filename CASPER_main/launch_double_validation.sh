#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/norman.json}"
CHECKPOINT_STEP="${2:-15000}"
SEED="${3:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python start_validate_double.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_STEP}" \
    --seed "${SEED}" \
    --output "results/norman_double_step${CHECKPOINT_STEP}.json"
