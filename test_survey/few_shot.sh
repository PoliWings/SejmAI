#!/bin/bash

set -e

EXTRA_ARGS=""
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        EXTRA_ARGS="--questions $1"
    else
        EXTRA_ARGS="$@"
    fi
fi

sides=("left" "right")
personas=("left" "right")

for side in "${sides[@]}"; do
    for persona in "${personas[@]}"; do
        echo "Running combination: Side=$side, Persona=$persona"
        python few_shot_prompting.py --side "$side" --persona "$persona" $EXTRA_ARGS
    done
done