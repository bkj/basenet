#!/usr/bin/env python

"""
    spnn.py
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

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-max', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    
    parser.add_argument('--wta-p', type=float, default=1.0)
    parser.add_argument('--wta-mode', type=str, default='per_image')
    parser.add_argument('--bn-disabled', action="store_true")
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--download', action="store_true")
    
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

BN_DISABLED = args.bn_disabled

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
    "test"  : testloader,
}

# --
# Model definition
# Derived from models in `https://github.com/kuangliu/pytorch-cifar`

class WTADropout(nn.Module):
    def __init__(self, p, mode='per_image'):
        super().__init__()
        
        self.mode = mode
        self.p    = p
        self.k    = 0
        
    def forward(self, x):
        bs = x.shape[0]
        nc = x.shape[1]
        sz = x.shape[2]
        assert x.shape[3] == sz
        
        # print(tuple(x.shape), file=sys.stderr)
        
        if self.mode == 'per_location':
            # take largest p% of activations per location per image
            
            k = int(np.ceil(nc * self.p))
            
            topk = x.topk(k=k, dim=1)[0]
            topk = topk[:,-1:,:,:]
            
        elif self.mode == 'per_channel':
            # take largest p% of activations per channel per image
            
            k = int(np.ceil(sz * self.p))
            
            tmp  = x.view(bs, nc, -1)
            topk = tmp.topk(k=k, dim=-1)[0]
            topk = topk[:,:,-1:]
            topk = topk.view(bs, nc, 1, 1)
        elif self.mode == 'per_image':
            # take largest p% of activations per image
            
            k = int(np.ceil(nc * sz * sz * self.p))
            
            tmp = x.view(bs, -1)
            topk = tmp.topk(k=k, dim=-1)[0]
            topk = topk[:,-1:]
            topk = topk.view(bs, 1, 1, 1)
        else:
            raise NotImplemented
        
        x = x * (x >= topk).float()
        
        if self.k:
            assert k == self.k
        else:
            self.k = k
        
        return x
    
    def __repr__(self):
        return 'WTADropout(p=%f | k=%d)' % (self.p, self.k if self.k else -1)


class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.wta1 = WTADropout(p=args.wta_p, mode=args.wta_mode)
        self.wta2 = WTADropout(p=args.wta_p, mode=args.wta_mode)
        
        # self.pre_nnz  = 0
        # self.post_nnz = 0
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                WTADropout(p=args.wta_p, mode=args.wta_mode), # !!
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(out)
        # self.pre_nnz += int((out != 0).sum())
        out = self.wta1(out) # !!
        # self.post_nnz += int((out != 0).sum())
        
        out = self.conv2(F.relu(self.bn2(out)))
        # self.pre_nnz += int((out != 0).sum())
        out = self.wta2(out)
        # self.post_nnz += int((out != 0).sum())
        
        return out + shortcut


class ResNet18(BaseNet):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__(loss_fn=F.cross_entropy)
        
        self.in_channels = 64
        
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 512, num_blocks[3], stride=2),
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.prep(x)
        x = self.layers(x)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --
# Define model

print('cifar10.py: initializing model...', file=sys.stderr)

cuda = torch.device('cuda')
model = ResNet18().to(cuda)
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
    lr_scheduler = getattr(HPSchedule, args.lr_schedule)(hp_max=args.lr_max, epochs=args.epochs + 1)

model.init_optimizer(
    opt=torch.optim.SGD,
    params=model.parameters(),
    hp_scheduler={"lr" : lr_scheduler},
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    nesterov=True,
)

_ = model.train_epoch(dataloaders, mode='train', num_batches=1, metric_fns=['n_correct'])
print(model, file=sys.stderr)

# --
# Train

print('cifar10.py: training...', file=sys.stderr)
t = time()
for epoch in range(args.epochs):
    train = model.train_epoch(dataloaders, mode='train', metric_fns=['n_correct'])
    # print([[layer.post_nnz / layer.pre_nnz for layer in layers] for layers in model.layers], file=sys.stderr)
    test  = model.eval_epoch(dataloaders, mode='test', metric_fns=['n_correct'])
    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.hp['lr'],
        "test_acc"  : float(test['acc']),
        "train_acc" : float(train['acc']),
        "time"      : time() - t,
    }))
    sys.stdout.flush()

