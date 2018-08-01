#!/bin/bash

# run.sh

mkdir -p results

# --
# DAWN submission

CUDA_VISIBLE_DEVICES=0 python cifar10.py --run linear50 --epochs 50 --lr-schedule linear --lr-max 0.1 > results/cifar10-linear-e50.jl
CUDA_VISIBLE_DEVICES=1 python cifar10.py --run one_cycle20 --epochs 20 --lr-schedule one_cycle --lr-max 0.1 > results/cifar10-one_cycle-e20.jl