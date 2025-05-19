#!/bin/bash

BASE_MODEL="Qwen/Qwen3-4B"

python train_local.py --data-path ./sft/right_model_sft.json --base-model $BASE_MODEL && \
python train_local.py --data-path ./sft/left_model_sft.json --base-model $BASE_MODEL