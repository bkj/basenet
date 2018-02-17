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
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()

# --
# From `fastai`

def get_children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())

# def apply_leaf(model, fn):
#     children = get_children(model)
#     if isinstance(model, nn.Module):
#         fn(model)
    
#     if len(children) > 0:
#         for layer in children:
#             apply_leaf(layer, fn)

# def _set_freeze(x, val):
#     p.frozen = val
#     for p in x.parameters():
#         p.requires_grad = val

# def set_freeze(model, val):
#     apply_leaf(model, lambda x: _set_freeze(x, val))

def set_freeze(x, mode):
    x.frozen = mode
    for p in x.parameters():
        p.requires_grad = not mode
    
    for module in x.children():
        set_freeze(module, mode)




