#!/usr/bin/env python

"""
    make-experiments.py
"""

import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

ranges = {
    "ac_m" : (0, 0.5),
    "ac_s" : (0, 6),
}

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    
    outdir = os.path.join('results', 'seed_%s' % args.seed)
    if os.path.exists(outdir):
        raise Exception('%s already exists!' % outdir)
    
    os.makedirs(outdir)
    
    exps = []
    for _ in range(args.n):
        ac_m = np.random.uniform(ranges['ac_m'][0], ranges['ac_m'][1], 1)
        ac_s = 2 ** np.random.uniform(ranges['ac_s'][0], ranges['ac_s'][1], 1)
        exps.append(
            'python cifar_additive_cosine.py --ac-m %f --ac-s %f > %s/m%f-s%f.jl' % 
                (ac_m, ac_s, outdir, ac_m, ac_s)
        )
    
    exps = list(set(exps))
    for exp in exps:
        print(exp)