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
    parser.add_argument('--arch', type=str, default='orig')
    
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--extra', type=int, default=5)
    parser.add_argument('--burnout', type=int, default=5)
    parser.add_argument('--lr-schedule', type=str, default='one_cycle')
    parser.add_argument('--lr-max', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--drop-path-prob', type=float, default=0.2)
    
    parser.add_argument('--sgdr-period-length', type=int, default=10)
    parser.add_argument('--sgdr-t-mult', type=int, default=2)
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--download', action="store_true")
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

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=512,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

dataloaders = {
    "train" : trainloader,
    "test"  : testloader,
}

# --
# Define model

print('cifar10.py: initializing model...', file=sys.stderr)

from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

if args.arch == 'orig':
    genotype = Genotype(
        normal=[
            ('sep_conv_3x3', 1), 
            ('sep_conv_3x3', 0), 
            ('skip_connect', 0), 
            ('sep_conv_3x3', 1), 
            ('skip_connect', 0), 
            ('sep_conv_3x3', 1), 
            ('sep_conv_3x3', 0), 
            ('skip_connect', 2)
        ],
        normal_concat=[2, 3, 4, 5],
        reduce=[
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)
        ],
        reduce_concat=[2, 3, 4, 5]
    )
elif args.arch == 'hyper':
    genotype = Genotype(
        normal=[
            ("skip_connect", 1),
            ("sep_conv_3x3", 0),
            ("sep_conv_5x5", 0),
            ("max_pool_3x3", 1),
            ("sep_conv_3x3", 2),
            ("max_pool_3x3", 1),
            ("sep_conv_5x5", 3),
            ("dil_conv_3x3", 0),
        ],
        normal_concat=[2, 3, 4, 5],
        reduce=[
            ("max_pool_3x3", 0),
            ("sep_conv_3x3", 1),
            ("skip_connect", 2),
            ("dil_conv_3x3", 0),
            ("dil_conv_5x5", 2),
            ("max_pool_3x3", 0),
            ("dil_conv_3x3", 4),
            ("skip_connect", 0)
        ],
        reduce_concat=[2, 3, 4, 5],
    )
else:
    raise Exception

cuda = 'cuda' # torch.device('cuda')
model = DARTModel(
    C=36,
    num_classes=10,
    layers=20,
    auxiliary=False,
    genotype=genotype
).to(cuda)
model.verbose = True
print(model, file=sys.stderr)

# --
# Initialize optimizer

print('cifar10.py: initializing optimizer...', file=sys.stderr)

if args.lr_schedule == 'linear_cycle':
    lr_scheduler = HPSchedule.linear_cycle(hp_max=args.lr_max, epochs=args.epochs, extra=args.extra)
elif args.lr_schedule == 'sgdr':
    lr_scheduler = HPSchedule.sgdr(
        hp_init=args.lr_max,
        period_length=args.sgdr_period_length,
        t_mult=args.sgdr_t_mult,
    )
else:
    lr_scheduler = getattr(HPSchedule, args.lr_schedule)(hp_max=args.lr_max, epochs=args.epochs)

model.init_optimizer(
    opt=torch.optim.SGD,
    params=model.parameters(),
    hp_scheduler={"lr" : lr_scheduler},
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    nesterov=True,
)

# --
# Train

print('cifar10.py: training...', file=sys.stderr)
t = time()
for epoch in range(args.epochs):
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    
    train = model.train_epoch(dataloaders, mode='train', compute_acc=True)
    test  = model.eval_epoch(dataloaders, mode='test', compute_acc=True)
    print(json.dumps({
        "epoch"          : int(epoch),
        "lr"             : model.hp['lr'],
        "test_acc"       : float(test['acc']),
        "train_acc"      : float(train['acc']),
        "drop_path_prob" : float(model.drop_path_prob),
        "time"           : time() - t,
    }))
    sys.stdout.flush()

model.save('weights')