#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/schmidt.json}"
CHECKPOINT_STEP="${2:-20000}"
SEED="${3:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python start_validate.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_STEP}" \
    --seed "${SEED}" \
    --output "results/$(basename "${CONFIG_PATH}" .json)_step${CHECKPOINT_STEP}.json"
