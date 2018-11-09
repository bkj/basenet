#!/bin/bash

# run.sh

mkdir -p _results
CUDA_VISIBLE_DEVICES=0 python main.py --train-size 1.0 > _results/hyperdart-0.jl
CUDA_VISIBLE_DEVICES=0 python main.py --train-size 0.8 > _results/hyperdart-1-trainsize0.8.jl