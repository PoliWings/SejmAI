#!/bin/bash

python train.py --data_path ./sft/right_model_sft.json && \
python train.py --data_path ./sft/left_model_sft.json