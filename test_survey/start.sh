#!/bin/bash

version="$1"

if [ -z "$version" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

for side in right left; do
    python fine_tuning.py --load-lora "$side" --version "$version" && \
    python model_testing.py --lora "$side"
done