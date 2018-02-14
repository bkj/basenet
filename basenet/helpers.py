#!/usr/bin/env python

"""
    helpers.py
"""

from __future__ import print_function, division

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def set_seeds(seed=100):
    np.random.seed(seed)
    _ = torch.manual_seed(seed + 123)
    _ = torch.cuda.manual_seed(seed + 456)