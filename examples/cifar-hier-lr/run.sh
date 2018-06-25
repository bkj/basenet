#!/bin/bash

# run.sh

CUDA_VISIBLE_DEVICES=0 ./run-one.sh constant &
CUDA_VISIBLE_DEVICES=1 ./run-one.sh step &
CUDA_VISIBLE_DEVICES=2 ./run-one.sh linear &
CUDA_VISIBLE_DEVICES=3 ./run-one.sh cosine &