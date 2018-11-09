#!/usr/bin/env python

"""
    cifar10.py
"""

from __future__ import division, print_function

import sys
import json
import argparse
import numpy as np
from time import time
from PIL import Image
from sklearn.model_selection import train_test_split

from basenet import BaseNet
from basenet.hp_schedule import HPSchedule
from basenet.helpers import to_numpy, set_seeds
from basenet.vision import transforms as btransforms

import torch
from torch import nn
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True

from torchvision import transforms, datasets

from model import NetworkCIFAR as DARTModel

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-max', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--download', action="store_true")
    parser.add_argument('--train-size', type=float, default=1.0)
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

# --
# IO

print('cifar10.py: making dataloaders...', file=sys.stderr)

transform_train = transforms.Compose([
    btransforms.ReflectionPadding(margin=(4, 4)),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    btransforms.NormalizeDataset(dataset='cifar10'),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    btransforms.NormalizeDataset(dataset='cifar10'),
])

try:
    trainset = datasets.CIFAR10(root='./data', train=True, download=args.download, transform=transform_train)
    testset  = datasets.CIFAR10(root='./data', train=False, download=args.download, transform=transform_test)
except:
    raise Exception('cifar10.py: error loading data -- try rerunning w/ `--download` flag')

train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=args.train_size, random_state=args.seed)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
    num_workers=4,
    pin_memory=True,
)

valloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
    num_workers=4,
    pin_memory=True,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)



dataloaders = {
    "train" : trainloader,
    "val"   : valloader,
    "test"  : testloader,
}

# --
# Define model

print('cifar10.py: initializing model...', file=sys.stderr)

# >>>>>>>>>>>>>>>>

from collections import namedtuple
from torch.autograd import Variable
from hyperband import HyperBand
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

MAX_EPOCHS = 32

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

def random_alphas(_steps=4):
    k = sum(1 for i in range(_steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    alphas_normal = Variable(1e-3*torch.randn(k, num_ops), requires_grad=False)
    alphas_reduce = Variable(1e-3*torch.randn(k, num_ops), requires_grad=False)
    return alphas_normal, alphas_reduce


def random_genotype(alphas_normal, alphas_reduce, _steps=4, _multiplier=4):
    
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(_steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    
    gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())
    
    concat = list(range(2+_steps-_multiplier, _steps+2))
    return Genotype(
      normal=gene_normal,
      normal_concat=concat,
      reduce=gene_reduce,
      reduce_concat=concat
    )


class HyperBandWrapper:
    def __init__(self, model, genotype):
        self.model = model
        self.genotype = genotype
        self.epoch = 0
    
    def train_until(self, epochs, config):
        while self.epoch < epochs:
            train = self.model.train_epoch(dataloaders, mode='train', compute_acc=True)
            valid = self.model.eval_epoch(dataloaders, mode='val', compute_acc=True)
            test  = self.model.eval_epoch(dataloaders, mode='test', compute_acc=True)
            print(json.dumps({
                "config"    : tuple(config),
                "epoch"     : int(self.epoch),
                "lr"        : self.model.hp['lr'],
                "valid_err" : 1 - float(valid['acc']),
                "test_err"  : 1 - float(test['acc']),
                "train_err" : 1 - float(train['acc']),
                "genotype"  : self.genotype,
            }))
            sys.stdout.flush()
            self.epoch += 1
        
        return 1 - float(valid['acc'])
    
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)


def random_dart_model(init_channels=16, layers=8):
    alphas_normal, alphas_reduce = random_alphas()
    genotype = random_genotype(alphas_normal, alphas_reduce)
    
    model = DARTModel(
        C=init_channels,
        num_classes=10,
        layers=layers,
        auxiliary=False,
        genotype=genotype
    )
    model.verbose = True
    model.drop_path_prob = 0
    
    model.init_optimizer(
        opt=torch.optim.SGD,
        params=model.parameters(),
        hp_scheduler={
            "lr" : HPSchedule.cyclical(hp_max=args.lr_max, epochs=MAX_EPOCHS),
        },
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    
    return HyperBandWrapper(model=model, genotype=genotype)

class Trainer:
    def __init__(self, rand_worker_fn):
        self.configs = {}
        self.rand_worker_fn = rand_worker_fn
    
    def random_configs(self, s, n):
        for nn in range(n):
            self.configs[(s, nn)] = self.rand_worker_fn()
        
        return [(s, nn) for nn in range(n)]
    
    def eval_config(self, config, iters):
        """ Evaluates model w/ given configuration on validation data """
        model = self.configs[config]
        _ = model.to('cuda')
        obj = model.train_until(epochs=iters, config=config)
        _ = model.to('cpu')
        return {
            "config"    : config,
            "obj"       : obj,
            "converged" : False,
        }


trainer = Trainer(rand_worker_fn=random_dart_model)
h = HyperBand(trainer, max_iter=MAX_EPOCHS, eta=2)
h.run()