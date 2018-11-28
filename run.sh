#!/bin/bash

# run.sh

# --
# Synthetic experiments

python synth.py

CUDA_VISIBLE_DEVICES=7 python synth.py
