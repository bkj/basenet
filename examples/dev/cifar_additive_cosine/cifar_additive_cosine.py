#!/usr/bin/env python

"""
    cifar_additive_cosine.py
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
    parser.add_argument('--extra', type=int, default=5)
    parser.add_argument('--burnout', type=int, default=5)
    parser.add_argument('--lr-schedule', type=str, default='linear_cycle')
    parser.add_argument('--lr-max', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    
    parser.add_argument('--sgdr-period-length', type=int, default=10)
    parser.add_argument('--sgdr-t-mult', type=int, default=2)

    parser.add_argument('--ac-m', type=float, default=0.0)
    parser.add_argument('--ac-s', type=float, default=1.0)
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--download', action="store_true")
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

# --
# IO

print('cifar_additive_cosine.py: making dataloaders...', file=sys.stderr)

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

class CustomCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return (img, target), target

try:
    trainset = CustomCIFAR10(root='./data', train=True, download=args.download, transform=transform_train)
    testset  = CustomCIFAR10(root='./data', train=False, download=args.download, transform=transform_test)
except:
    raise Exception('cifar_additive_cosine.py: error loading data -- try rerunning w/ `--download` flag')

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
# Model definition

class NormLinear(nn.Linear):
    def __init__(self, *args, ac_m=0.0, ac_s=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ac_m = ac_m
        self.ac_s = ac_s
    
    def forward(self, x, y):
        sim = F.linear(
            F.normalize(x, dim=-1),
            self.weight / ((self.weight ** 2).sum(dim=-1).sqrt().unsqueeze(-1) + 1e-5),
            self.bias,
        )
        
        if self.training:
            y = y.unsqueeze(-1)
            sim.scatter_(1, y, sim.gather(1, y) - self.ac_m)
        
        return self.ac_s * sim


class PreActBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet18(BaseNet):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10, ac_m=0, ac_s=1):
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
            self._make_layer(256, 256, num_blocks[3], stride=2),
        )
        
        self.classifier = NormLinear(512, num_classes, bias=False, ac_m=ac_m, ac_s=ac_s)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, data):
        x, y = data
        x = x.half()
        x = self.prep(x)
        
        x = self.layers(x)
        
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)
        
        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)
        
        x = torch.cat([x_avg, x_max], dim=-1)
        
        x = self.classifier(x, y)
        
        return x

# --
# Define model

print('cifar_additive_cosine.py: initializing model...', file=sys.stderr)

cuda = torch.device('cuda')
model = ResNet18(ac_m=args.ac_m, ac_s=args.ac_s).to(cuda).half()
model.verbose = True
print(model, file=sys.stderr)

# --
# Initialize optimizer

print('cifar_additive_cosine.py: initializing optimizer...', file=sys.stderr)

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

print('cifar_additive_cosine.py: training...', file=sys.stderr)
t = time()
for epoch in range(args.epochs + args.extra + args.burnout):
    train = model.train_epoch(dataloaders, mode='train')
    test  = model.eval_epoch(dataloaders, mode='test')
    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.hp['lr'],
        "test_acc"  : float(test['acc']),
        "train_acc" : float(train['acc']),
        "time"      : time() - t,

        "ac_m"      : args.ac_m,
        "ac_s"      : args.ac_s,
    }))
    sys.stdout.flush()

model.save('weights')