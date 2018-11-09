#!/bin/bash

# run.sh

mkdir -p _results
CUDA_VISIBLE_DEVICES=0 python main.py > _results/fastdart-0.jl

# --
# 

CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr-schedule linear \
    --epochs 50 > _results/dart-linear-50.jl

CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr-schedule linear \
    --epochs 100 > _results/dart-linear-100.jl

# --
# How big of a difference is droppath making?
# ... some ...

CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr-schedule linear \
    --drop-path-prob 0.0 \
    --epochs 50 > _results/dart-linear-50-nodroppath.jl

CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr-schedule linear \
    --drop-path-prob 0.0 \
    --epochs 100 > _results/dart-linear-100-nodroppath.jl

# --
#

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch hyper \
    --lr-schedule linear \
    --epochs 50 > _results/hyperdart-linear-50.jl

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch hyper \
    --lr-schedule linear \
    --drop-path-prob 0.0 \
    --epochs 50 > _results/hyperdart-linear-50-nodroppath.jl

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch hyper \
    --lr-schedule linear \
    --epochs 100 > _results/hyperdart-linear-100.jl

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch hyper \
    --lr-schedule linear \
    --drop-path-prob 0.0 \
    --epochs 100 > _results/hyperdart-linear-100-nodroppath.jl