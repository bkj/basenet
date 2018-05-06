#!/bin/bash

# runs.h

python cifar_opt.py --epochs 30 > results/cifar_opt-30.jl

# CUDA_VISIBLE_DEVICES=1 python cifar_opt2.py > cifar_opt2.jl
