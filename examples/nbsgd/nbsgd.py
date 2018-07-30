#!/usr/bin/env python

"""
    nbsgd.py
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils import data

from basenet import BaseNet
from basenet.hp_schedule import HPSchedule
from basenet.helpers import to_numpy, set_seeds

# --
# Helpers

def calc_r(y_i, x, y):
    x = x.sign()
    p = x[np.argwhere(y == y_i)[:,0]].sum(axis=0) + 1
    q = x[np.argwhere(y != y_i)[:,0]].sum(axis=0) + 1
    p, q = np.asarray(p).squeeze(), np.asarray(q).squeeze()
    return np.log((p / p.sum()) / (q / q.sum()))

# --
# IO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-train', type=str, default='./data/aclImdb/X_train.npy')
    parser.add_argument('--x-train-words', type=str, default='./data/aclImdb/X_train_words.npy')
    parser.add_argument('--y-train', type=str, default='./data/aclImdb/y_train.npy')
    
    parser.add_argument('--x-test', type=str, default='./data/aclImdb/X_test.npy')
    parser.add_argument('--x-test-words', type=str, default='./data/aclImdb/X_test_words.npy')
    parser.add_argument('--y-test', type=str, default='./data/aclImdb/y_test.npy')
    
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr-max', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=256)
    
    parser.add_argument('--vocab-size', type=int, default=200000)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

# --
# IO

print('nbsgd.py: making dataloaders...', file=sys.stderr)

X_train       = np.load(args.x_train).item()
X_train_words = np.load(args.x_train_words).item()
y_train       = np.load(args.y_train)

X_test        = np.load(args.x_test).item()
X_test_words  = np.load(args.x_test_words).item()
y_test        = np.load(args.y_test)

train_dataset = torch.utils.data.dataset.TensorDataset(
    torch.from_numpy(X_train_words.toarray()).long(),
    torch.from_numpy(y_train).long()
)
test_dataset = torch.utils.data.dataset.TensorDataset(
    torch.from_numpy(X_test_words.toarray()).long(),
    torch.from_numpy(y_test).long(),
)

dataloaders = {
    "train" : torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
    "test"  : torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
}

n_classes = int(y_train.max()) + 1

r = np.column_stack([calc_r(i, X_train, y_train) for i in range(n_classes)])
r = np.vstack([[0.0] * n_classes, r])
r = r[:,0]
# r = np.random.normal(0, 1e-1, r.shape)
r = torch.FloatTensor(r).cuda()

# --
# Model definition

class DotProdNB(BaseNet):
    def __init__(self, vocab_size, n_classes, r, w_adj=0.4, r_adj=10):
        
        super().__init__()
        
        # Init w
        self.w_weight    = torch.zeros(vocab_size + 1, 1).uniform_(-0.1, 0.1)
        self.w_weight[0] = 0
        self.w_weight    = nn.Parameter(self.w_weight)
        
        self.r_weight = torch.stack([r, -r], dim=1)
        
        self.w_adj = w_adj
        self.r_adj = r_adj
        
    def forward(self, feat_idx):
        n_docs, n_words = feat_idx.shape
        
        feat_idx  = feat_idx.view(-1)
        zero_mask = (feat_idx == 0)
        
        w = (self.w_weight[feat_idx] + self.w_adj)
        w[zero_mask] = 0
        w = w.view(n_docs, n_words, self.w_weight.shape[1])
        
        r = self.r_weight[feat_idx]
        r[zero_mask] = 0
        r = r.view(n_docs, n_words, self.r_weight.shape[1])
        
        x = (w * r).sum(dim=1)
        x =  x / self.r_adj
        return x

# --
# Define model

print('nbsgd.py: initializing model...', file=sys.stderr)

model = DotProdNB(args.vocab_size, n_classes, r).to(torch.device('cuda'))
model.verbose = True
print(model, file=sys.stderr)

# --
# Initializing optimizer 

lr_scheduler = HPSchedule.constant(hp_max=0.5)
model.init_optimizer(
    opt=torch.optim.SGD,
    params=[p for p in model.parameters() if p.requires_grad],
    hp_scheduler={
        "lr" : lr_scheduler
    },
    momentum=0.99,
)

# --
# Train

print('nbsgd.py: training...', file=sys.stderr)
t = time()
for epoch in range(args.epochs):
    train = model.train_epoch(dataloaders, mode='train', compute_acc=True)
    test  = model.eval_epoch(dataloaders, mode='test', compute_acc=True)
    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.hp['lr'],
        "test_acc"  : float(test['acc']),
        "time"      : time() - t,
    }))
    sys.stdout.flush()

model.save('weights')
