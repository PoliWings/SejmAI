#!/bin/bash

run_with_retries() {
    local max_retries=5
    local attempt=1
    local cmd=("$@")

    while [ $attempt -le $max_retries ]; do
        echo "----------------------------------------------------"
        echo "Running (Attempt $attempt/$max_retries): ${cmd[*]}"
        
        "${cmd[@]}"
        
        if [ $? -eq 0 ]; then
            return 0
        fi

        echo "Error: Command failed."
        if [ $attempt -lt $max_retries ]; then
            echo "Retrying in 3 seconds..."
            sleep 3
        fi
        ((attempt++))
    done

    echo "Critical Error: Failed to execute command after $max_retries attempts."
    return 1
}

args=("$@")
version=""
pass_args=()
model_name=""
dataset=""

for ((i=0; i<${#args[@]}; i++)); do
    arg="${args[i]}"
    if [[ "$arg" == "--version" ]]; then
        ((i++))
        version="${args[i]}"
    elif [[ "$arg" == "--model-name" ]]; then
        ((i++))
        model_name="${args[i]}"
    elif [[ "$arg" == "--dataset" ]]; then
        ((i++))
        dataset="${args[i]}"
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
if [[ -n "$dataset" ]]; then
    filtered_args+=(--dataset "$dataset")
fi

if [[ "$mode" == "service" ]]; then
    python ../fine_tuning/train_service.py --unload-lora left
    python ../fine_tuning/train_service.py --unload-lora right
    
    run_with_retries python model_testing.py --service "${filtered_args[@]}" || exit 1

    for side in right left; do
        echo "=== Load lora adapter for $side side ==="
        python ../fine_tuning/train_service.py --load-lora "$side" "${version_arg[@]}" || exit 1

        echo "=== Testing model with $side adapter ==="
        run_with_retries python model_testing.py --side "$side" --service "${filtered_args[@]}" || exit 1
        
        python ../fine_tuning/train_service.py --unload-lora "$side"
    done

elif [[ "$mode" == "local" ]]; then
    run_with_retries python model_testing.py "${filtered_args[@]}" || exit 1
    
    for side in right left; do
        echo "=== Testing local model with $side version ==="
        run_with_retries python model_testing.py --side "$side" "${filtered_args[@]}" || exit 1
    done

else
    echo -e "Usage: $0 (--service --version <version> | --local --model-name <model>)"
    echo -e "\tmode:\t\t--service or --local"
    echo -e "\tversion:\trequired only for --service"
    echo -e "\tmodel-name:\trequired only for --local"
    echo -e "\tdataset:\toptional, specify dataset name"
    echo -e "\tflags:\t\tany additional flags passed to model_testing.py"
    echo -e "\tExample:\t$0 --service --version 1.0.0 --dataset my_dataset"
    echo -e "\t\t\t$0 --local --model-name my_model --dataset my_dataset"
    exit 1
fi