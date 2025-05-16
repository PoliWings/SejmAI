#!/bin/bash

# python model_testing.py --local right && \
# python model_testing.py --local left

python model_testing.py --lora right &&
python model_testing.py --lora left