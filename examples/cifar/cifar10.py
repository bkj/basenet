#!/usr/bin/env python

"""
    cifar10.py
    
    Train preactivation ResNet18 on CIFAR10 w/ linear learning rate annealing
    
    After 50 epochs:
        {"epoch": 49, "lr": 5.115089514063697e-06, "test_loss": 0.3168042216449976, "test_acc": 0.9355}
"""

from __future__ import division, print_function

import sys
import json
import argparse
import numpy as np

from basenet import BaseNet
from basenet.lr import LRSchedule
from basenet.helpers import to_numpy, set_seeds

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

from torchvision import transforms, datasets

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--download', action="store_true")
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

# --
# IO

print('cifar10.py: making dataloaders...', file=sys.stderr)

cifar10_stats = {
    "mean" : (0.4914, 0.4822, 0.4465),
    "std"  : (0.24705882352941178, 0.24352941176470588, 0.2615686274509804),
}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
])

try:
    trainset = datasets.CIFAR10(root='./data', train=True, download=args.download, transform=transform_train)
    testset  = datasets.CIFAR10(root='./data', train=False, download=args.download, transform=transform_test)
except:
    raise Exception('cifar10.py: error loading data -- try rerunning w/ `--download` flag')

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

dataloaders = {
    "train" : trainloader,
    "test"  : testloader,
}

# --
# Model definition
# Derived from models in `https://github.com/kuangliu/pytorch-cifar`

class PreActBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        
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
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet18, self).__init__(loss_fn=F.cross_entropy)
        
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
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# --
# Define model

print('cifar10.py: initializing model...', file=sys.stderr)

model = ResNet18().cuda()
print(model, file=sys.stderr)
model.verbose = True

# --
# Initialize optimizer

print('cifar10.py: initializing optimizer...', file=sys.stderr)

lr_scheduler = getattr(LRSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs)
model.init_optimizer(
    opt=torch.optim.SGD,
    params=model.parameters(),
    lr_scheduler=lr_scheduler,
    momentum=0.9,
    weight_decay=5e-4,
)

# --
# Train

print('cifar10.py: training...', file=sys.stderr)
for epoch in range(args.epochs):
    train = model.train_epoch(dataloaders, mode='train')
    test  = model.eval_epoch(dataloaders, mode='test')
    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.lr,
        "train_acc" : float(train['acc']),
        "test_acc"  : float(test['acc']),
    }))
    sys.stdout.flush()
