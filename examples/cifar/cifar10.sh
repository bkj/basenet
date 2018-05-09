#!/bin/bash

# cifar10.sh

mkdir -p results

CUDA_VISIBLE_DEVICES=1 python cifar10.py \
    --epochs 630 \
    --extra 0 \
    --burnout 0 \
    --lr-schedule sgdr \
    --lr-max 0.1 \
    --drop-path-prob 0.2 \
    --seed 123 > results/cifar10-sgdr-630-dp-0.2.jl


CUDA_VISIBLE_DEVICES=0 python cifar10.py \
    --epochs 630 \
    --extra 0 \
    --burnout 0 \
    --lr-schedule sgdr \
    --lr-max 0.1 \
    --drop-path-prob 0.4 \
    --seed 123 > results/cifar10-sgdr-630-dp-0.4.jl
