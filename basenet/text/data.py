#!/usr/bin/env python

"""
    data.py
"""

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SequentialSampler

class RaggedDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), 'len(X) != len(y)'
        self.X = [torch.LongTensor(xx) for xx in X]
        self.y = torch.LongTensor(y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)


def text_collate_fn(batch, pad_value=1):
    X, y = zip(*batch)
    
    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value) for xx in X]
    
    X = torch.stack(X, dim=-1)
    y = torch.LongTensor(y)
    return X, y