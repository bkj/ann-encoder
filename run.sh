#!/bin/bash

# run.sh

# --
# Synthetic experiments

python synth.py

CUDA_VISIBLE_DEVICES=7 python synth.py

# --
# ML20 experiments

python train.py --cache-path cache/ml20
python inference.py --cache-path cache/ml20