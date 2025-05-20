#!/bin/bash

args=("$@")
version=""
pass_args=()
model_name=""

for ((i=0; i<${#args[@]}; i++)); do
    arg="${args[i]}"
    if [[ "$arg" == "--version" ]]; then
        ((i++))
        version="${args[i]}"
    elif [[ "$arg" == "--model-name" ]]; then
        ((i++))
        model_name="${args[i]}"
    else
        pass_args+=("$arg")
    fi
done

mode=""

for arg in "${pass_args[@]}"; do
    if [[ "$arg" == "--local" ]]; then
        mode="local"
        break
    elif [[ "$arg" == "--service" ]]; then
        mode="service"
        break
    fi
done

filtered_args=()
if [[ -n "$model_name" ]]; then
    filtered_args+=(--model-name "$model_name")
fi
if [[ -n "$version" ]]; then
    version_arg=(--version "$version")
fi

if [[ "$mode" == "service" ]]; then
    python model_testing.py --service "${filtered_args[@]}"

    for side in right left; do
        echo "=== Load lora adapter for $side side ==="
        python ../fine_tuning/train_service.py --load-lora "$side" "${version_arg[@]}" || exit 1

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
    echo -e "Usage: $0 (--service --version <version> | --local --model-name <model>)"
    echo -e "\tmode:\t\t--service or --local"
    echo -e "\tversion:\trequired only for --service"
    echo -e "\tmodel-name:\trequired only for --local"
    echo -e "\tflags:\t\tany additional flags passed to model_testing.py"
    echo -e "\tExample:\t$0 --service --version 1.0.0"
    echo -e "\t\t\t$0 --local --model-name my_model"
    exit 1
fi
