#!/bin/bash

# run.sh

mkdir -p results

# --
# Run

# Train w/ BN
# !! This has been manually tuned previously
time CUDA_VISIBLE_DEVICES=7 \
    python baseline.py --lr-max 0.1 > results/baseline-bn.jl

# Train w/o BN
# !! This has not been manually tuned, so we need to do a search
#       at some point
time CUDA_VISIBLE_DEVICES=7 \
    python baseline.py --lr-max 0.01 --bn-disabled > results/baseline-nobn.jl


function run_spnn {
    WTA_P=$1
    time CUDA_VISIBLE_DEVICES=7 \
        python spnn.py --lr-max 0.01 --wta-p $WTA_P > results/spnn-p$WTA_P.jl
}

run_spnn 0.1
run_spnn 0.2
run_spnn 0.3
run_spnn 0.4
run_spnn 0.5
run_spnn 0.6
run_spnn 0.7
run_spnn 0.8
run_spnn 0.9
run_spnn 1.0

run_spnn 0.01
run_spnn 0.02
run_spnn 0.03
run_spnn 0.04
run_spnn 0.05
run_spnn 0.06
run_spnn 0.07
run_spnn 0.08
run_spnn 0.09
echo "done"

# per_channel + full activations if reduced is less than 10
# python spnn.py --lr-max 0.01 --wta-p 0.1 > results/spnn-p0.1-floor.jl

python spnn.py --lr-max 0.01 --wta-p 0.1 --wta-mode per_image

mkdir -p results/spnn/per_image/lr0.05
CUDA_VISIBLE_DEVICES=5 python spnn.py --lr-max 0.001 --wta-p 0.1 --wta-mode per_image > results/spnn/per_image/lr0.05/adam_p0.001.jl

CUDA_VISIBLE_DEVICES=6 python spnn.py --lr-max 0.1 --wta-p 0.5 --wta-mode per_image > results/spnn/per_image/lr0.1/p0.5.jl
CUDA_VISIBLE_DEVICES=6 python spnn.py --lr-max 0.1 --wta-p 0.2 --wta-mode per_image > results/spnn/per_image/lr0.1/p0.2.jl