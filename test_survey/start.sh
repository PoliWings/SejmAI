#!/bin/bash

args=("$@")
flags=""
mode=""
version=""

for arg in "${args[@]}"; do
    if [[ "$arg" == "service" || "$arg" == "local" ]]; then
        mode="$arg"
    elif [[ -z "$version" && "$arg" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        version="$arg"
    else
        flags+=" --$arg"
    fi
done

if [ "$mode" = "service" ]; then
    if [ -z "$version" ]; then
        echo "Usage: $0 service <version> [flags]"
        exit 1
    fi
    for side in right left; do
        python ../fine_tuning/fine_tuning.py --load-lora "$side" --version "$version" && \
        python model_testing.py --lora "$side" $flags
    done
elif [ "$mode" = "local" ]; then
    for side in right left; do
        python model_testing.py --local "$side" $flags
    done
else
    echo "Usage: $0 <mode> [version] [flags]"
    echo "mode: service or local"
    echo "version: version of the model to test (only for service mode)"
    echo "flags: additional optional flags: base"
    echo "Example: $0 service 1.0.0"
    echo "Example: $0 local base"
    exit 1
fi
