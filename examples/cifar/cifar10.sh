#!/bin/bash

# cifar10.sh

time CUDA_VISIBLE_DEVICES=1 cifar10.py --epochs 150 > cifar10-150.jl
time CUDA_VISIBLE_DEVICES=1 python cifar10.py --epochs 310 > cifar10-310.jl
time CUDA_VISIBLE_DEVICES=1 python cifar10.py --epochs 630 > cifar10-630.jl
echo done