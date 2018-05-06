#!/bin/bash

# cifar10.sh

mkdir -p results

python cifar10.py \
    --epochs 150 \
    --extra 0 \
    --burnout 0 \
    --lr-schedule sgdr \
    --lr-max 0.1 \
    --download \
    --seed 123 > results/cifar10-sgdr-150-dp.jl
