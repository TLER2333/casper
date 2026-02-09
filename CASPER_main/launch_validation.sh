#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

CHECKPOINT_STEP=10000
CONFIG_PATH="config.json"

python start_validation.py \
    --checkpoint $CHECKPOINT_STEP \
    --config $CONFIG_PATH