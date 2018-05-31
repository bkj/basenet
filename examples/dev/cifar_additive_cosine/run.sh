#!/bin/bash

# run.sh

mkdir -p results

export CUDA_VISIBLE_DEVICES=0
SEED=123
python make-experiments.py --seed $SEED > experiments-$SEED.sh
chmod +x experiments-0.sh

./experiments-$SEED.sh