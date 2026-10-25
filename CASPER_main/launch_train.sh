#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/schmidt.json}"
SEED="${2:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python start_train.py --config "${CONFIG_PATH}" --seed "${SEED}"
