#!/bin/bash

base_model=""

for ((i=0; i<${#@}; i++)); do
    if [[ "${!i}" == "--base-model" ]]; then
        next=$((i+1))
        base_model="${!next}"
        break
    fi
done

if [[ -z "$base_model" ]]; then
    echo "Usage: $0 --base-model <model-name>"
    exit 1
fi

python train_local.py --data-path ./sft/right_model_sft.json --base-model "$base_model" && \
python train_local.py --data-path ./sft/left_model_sft.json --base-model "$base_model"
