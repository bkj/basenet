#!/bin/bash

# cifar10.sh

mkdir -p results

python cifar10.py \
    --epochs 150 \
    --lr-schedule sgdr \
    --lr-init 0.1 \
    --download \
    --seed 123 > results/cifar10-linear.jl
