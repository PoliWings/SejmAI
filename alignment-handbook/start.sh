#!/bin/bash

# nohup ./start.sh &> out.log &
# ps -ef | grep python
# pkill -f "python"

# watch -n 0.5 -d free -h
# watch -n 0.5 -d nvidia-smi

# flash_attn-2.7.3+cu11torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py ../fine_tuning/bielik/left_sft_qlora.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py ../fine_tuning/bielik/right_sft_qlora.yaml