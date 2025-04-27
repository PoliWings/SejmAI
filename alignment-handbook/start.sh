#!/bin/bash

# Set common parameters
ACCEL_CFG="recipes/accelerate_configs/multi_gpu.yaml"
ACCELERATE_LOG_LEVEL=info
NUM_PROC=1

# Fine-tune LEFT
accelerate launch \
  --config_file $ACCEL_CFG \
  --num_processes $NUM_PROC \
  scripts/run_sft.py \
  ../fine_tuning/bielik/left_sft_qlora.yaml

# Fine-tune RIGHT
accelerate launch \
  --config_file $ACCEL_CFG \
  --num_processes $NUM_PROC \
  scripts/run_sft.py \
  ../fine_tuning/bielik/right_sft_qlora.yaml
