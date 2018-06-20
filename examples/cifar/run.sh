#!/bin/bash

# run.sh

mkdir -p results

# --
# DAWN submission

time python cifar10.py > results/cifar-dawn.jl

time python cifar10.py --epochs 30 --lr-schedule linear