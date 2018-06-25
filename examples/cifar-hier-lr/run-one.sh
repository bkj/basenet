#!/bin/bash

# run-one.sh

mkdir -p results

MACRO=$1
python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule constant --lr-max 0.15 > ./results/$MACRO-constant_15e-2.jl
python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule step --lr-max 0.15 > ./results/$MACRO-step_15e-2.jl
python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule linear --lr-max 0.15 > ./results/$MACRO-linear_15e-2.jl
python cifar10.py --lr-macro-schedule $MACRO --lr-micro-schedule cosine --lr-max 0.15 > ./results/$MACRO-cosine_15e-2.jl
