#!/bin/bash

# run.sh

python make_commands.py
chmod +x cmd.sh


python cifar10_distill.py --distillation-alpha 0.0