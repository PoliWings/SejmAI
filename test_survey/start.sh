#!/bin/bash

args=("$@")
version=""
pass_args=()
model_name=""

for ((i=0; i<${#args[@]}; i++)); do
    arg="${args[i]}"
    if [[ -z "$version" && "$arg" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        version="$arg"
    elif [[ "$arg" == "--model-name" ]]; then
        ((i++))
        model_name="${args[i]}"
    else
        pass_args+=("$arg")
    fi
done

mode=""
for arg in "${pass_args[@]}"; do
    if [[ "$arg" == "--service" ]]; then
        mode="service"
        break
    fi
done

if [[ -z "$mode" ]]; then
    mode="local"
fi

filtered_args=()
if [[ -n "$model_name" ]]; then
    filtered_args+=(--model-name "$model_name")
fi

if [[ "$mode" == "service" ]]; then
    if [ -z "$version" ]; then
        echo "Usage: $0 --service <version> [other flags]"
        exit 1
    fi

    python model_testing.py --service "${filtered_args[@]}"

    for side in right left; do
        echo "=== Fine-tuning for $side side ==="
        python ../fine_tuning/train_service.py --load-lora "$side" --version "$version" || exit 1

        echo "=== Testing model with $side adapter ==="
        python model_testing.py --side "$side" --service "${filtered_args[@]}"
    done

elif [[ "$mode" == "local" ]]; then
    python model_testing.py "${filtered_args[@]}"
    for side in right left; do
        echo "=== Testing local model with $side version ==="
        python model_testing.py --side "$side" "${filtered_args[@]}"
    done

else
    echo "Usage: $0 (--service <version> | --local --model-name <model>)"
    echo "mode: --service or --local"
    echo "version: required only for --service"
    echo "model-name: required only for --local"
    echo "flags: any additional flags passed to model_testing.py"
    echo "Example: $0 --service 1.0.0"
    echo "Example: $0 --local --model-name my_model"
    exit 1
fi
