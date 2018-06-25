#!/bin/bash

# run-one.sh

mkdir -p results

MACRO=$1

python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule constant --lr-max 2e-1 > ./results/$MACRO-constant_2e-1.jl
python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule step --lr-max 2e-1 > ./results/$MACRO-step_2e-1.jl
python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule linear --lr-max 2e-1 > ./results/$MACRO-linear_2e-1.jl
python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule cosine --lr-max 2e-1 > ./results/$MACRO-cosine_2e-1.jl