#!/bin/bash

# flash_attn-2.7.3+cu11torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py ../fine_tuning/bielik/sft_qlora.yaml