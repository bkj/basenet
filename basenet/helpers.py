#!/usr/bin/env python

"""
    helpers.py
"""

from __future__ import print_function, division

import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Variable

# --
# Utils

def set_seeds(seed=100):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 123)
    _ = torch.cuda.manual_seed(seed + 456)
    _ = random.seed(seed + 789)

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, Variable):
        return to_numpy(x.data)
    else:
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()

# --
# From `fastai`

def get_children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())

def set_freeze(x, mode):
    x.frozen = mode
    for p in x.parameters():
        p.requires_grad = not mode
    
    for module in x.children():
        set_freeze(module, mode)

def apply_init(m, init_fn):
    def _cond_init(m, init_fn):
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(m, 'weight'):
                init_fn(m.weight)
            
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0.)
    
    m.apply(lambda x: _cond_init(x, init_fn))

def get_num_features(model):
    children = get_children(model)
    if len(children) == 0:
        return None
    
    for layer in reversed(children):
        if hasattr(layer, 'num_features'):
            return layer.num_features
        
        res = get_num_features(layer)
        if res is not None:
            return res