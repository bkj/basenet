#!/bin/bash

# run.sh

mkdir -p results

export CUDA_VISIBLE_DEVICES=0

# --

# ranges = {
#     "ac_m" : (0, 0.5),
#     "ac_s" : (0, 6),
# }

SEED=123
python make-experiments.py --seed $SEED > experiments-$SEED.sh
chmod +x experiments-$SEED.sh

./experiments-$SEED.sh

# --

# ranges = {
#     "ac_m" : (0, 0.2),
#     "ac_s" : (3, 5),
# }

SEED=456
python make-experiments.py --seed $SEED > experiments-$SEED.sh
chmod +x experiments-$SEED.sh

./experiments-$SEED.sh

