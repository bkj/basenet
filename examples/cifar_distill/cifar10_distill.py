#!/usr/bin/env python

"""
    cifar10_distill.py
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
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

from torchvision import transforms, datasets

# --
# Helpers

class DistillationWrapper(object):
    def __init__(self, dataset, z):
        self.dataset = dataset
        self.z = z
        
        assert len(z) == len(dataset)
    
    def __getitem__(self, index):
        
        x, y = self.dataset[index]
        z = self.z[index]
        
        return x, (y, z)
    
    def __len__(self):
        return len(self.dataset)


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr-schedule', type=str, default='linear_cycle')
    parser.add_argument('--lr-max', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=789)
    parser.add_argument('--download', action="store_true")
    
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--distillation-alpha', type=float, default=0.5)
    
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

# Distillation targets
test_preds   = np.load('test_preds.npy')

test_targets = np.load('test_targets.npy')
test_preds = test_preds[~np.isnan(test_preds).any(axis=(1, 2))]
top_models = np.argsort((test_preds.argmax(axis=-1) == test_targets).mean(axis=-1))[::-1]

test_preds = test_preds[top_models[:30]].mean(axis=0)
testset = DistillationWrapper(testset, test_preds)

train_preds = np.load('train_preds.npy')
train_preds = train_preds[~np.isnan(train_preds).any(axis=(1, 2))]
train_preds = train_preds[top_models[:30]].mean(axis=0)
trainset = DistillationWrapper(trainset, train_preds)


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
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10, loss_fn=F.cross_entropy):
        super(ResNet18, self).__init__(loss_fn=loss_fn)
        
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
        
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)
        
        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)
        
        x = torch.cat([x_avg, x_max], dim=-1)
        
        x = self.classifier(x)
        
        return x

# --
# Define model

print('cifar10.py: initializing model...', file=sys.stderr)

def distillation_loss(alpha, T=1):
    def _f(X, y):
        log_X = F.log_softmax(X, dim=-1)
        
        hard_loss = F.nll_loss(log_X, y[0])
        
        y_soft_softmax = F.softmax(y[1] / T, dim=-1)
        soft_loss = - (y_soft_softmax * log_X).sum(dim=-1).mean()
        
        return alpha * hard_loss + (1 - alpha) * soft_loss
        
    return _f


loss_fn = distillation_loss(alpha=args.distillation_alpha)
device = torch.device('cuda')
model = ResNet18(loss_fn=loss_fn).to(device)
print(model, file=sys.stderr)

model.verbose = True

# --
# Initialize optimizer

print('cifar10.py: initializing optimizer...', file=sys.stderr)

lr_scheduler = getattr(HPSchedule, args.lr_schedule)(hp_max=args.lr_max, epochs=args.epochs)
model.init_optimizer(
    opt=torch.optim.SGD,
    params=model.parameters(),
    hp_scheduler={"lr" : lr_scheduler},
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    # nesterov=True,
)

# --
# Train

print('cifar10.py: training...', file=sys.stderr)
t = time()
for epoch in range(args.epochs):
    train = model.train_epoch(dataloaders, mode='train')
    test  = model.eval_epoch(dataloaders, mode='test')
    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.hp['lr'],
        "test_acc"  : float(test['acc']),
        "time"      : time() - t,
        
        "weight_decay" : float(args.weight_decay),
        "momentum"     : float(args.momentum),
        "alpha"        : float(args.distillation_alpha),
    }))
    sys.stdout.flush()

