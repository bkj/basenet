
import numpy as np
import itertools

CMD = (
    "python cifar10_distill.py"
    "    --weight-decay %f"
    "    --momentum %f"
    "    --distillation-alpha %f"
    " > results/%f-%f-%f"
)

weight_decays = [5e-4, 0]
momentums     = [0.99, 0.9, 0.8, 0.5, 0.2, 0.0]
alphas        = [1.0, 0.75, 0.5, 0.25, 0.0]

params = itertools.product(weight_decays, momentums, alphas)
params = np.random.permutation(list(params))

f = open('cmd.sh', 'w')
f.write('#!/bin/bash\n')
for weight_decay, momentum, alpha in params:
     cmd = CMD % ((weight_decay, momentum, alpha) + (weight_decay, momentum, alpha))
     print(cmd, file=f)

f.close()