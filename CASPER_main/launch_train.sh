#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
NUM_GPUS=1
CONFIG_PATH="data/config_norman.json"
MASTER_PORT=29500

echo "Starting training on $NUM_GPUS GPUs..."

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    start_train.py \
    --config $CONFIG_PATH