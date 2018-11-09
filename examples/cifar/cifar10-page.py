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

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--lr-max', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--download', action="store_true")
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

# --
# IO

print('cifar10.py: making dataloaders...', file=sys.stderr)

transform_train = transforms.Compose([
    # btransforms.ReflectionPadding(margin=(4, 4)),
    # transforms.RandomCrop(32),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    btransforms.NormalizeDataset(dataset='cifar10'),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    btransforms.NormalizeDataset(dataset='cifar10'),
])

try:
    raw_trainset = datasets.CIFAR10(root='./data', train=True, download=args.download, transform=transform_train)
    raw_testset  = datasets.CIFAR10(root='./data', train=False, download=args.download, transform=transform_test)
except:
    raise Exception('cifar10.py: error loading data -- try rerunning w/ `--download` flag')

# # >>
# print('prepping data', file=sys.stderr)
# X_train, y_train = zip(*raw_trainset)
# X_test, y_test   = zip(*raw_testset)

# X_train, X_test = torch.stack(X_train), torch.stack(X_test)
# y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

# # X_train, X_test, y_train, y_test = X_train.cuda(), X_test.cuda(), y_train.cuda(), y_test.cuda()

# trainset = torch.utils.data.TensorDataset(X_train, y_train)
# testset  = torch.utils.data.TensorDataset(X_test, y_test)
# # <<


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

dataloaders = {
    "train" : trainloader,
    "test"  : testloader,
}

# --
# Model definition
# Derived from models in `https://github.com/kuangliu/pytorch-cifar`

# class HalfBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.half()

# class PreActBlock(nn.Module):
    
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
        
#         self.bn1   = HalfBatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2   = HalfBatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 # !! Modified per David Page
#                 # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.MaxPool2d(kernel_size=2, stride=2)
#             )
            
#     def forward(self, x):
#         if hasattr(self, 'shortcut'):
#             # return x
#             return self.shortcut(x)
#             # return self.shortcut(F.relu(self.bn1(x)))
#         else:
#             return x
        
#         # out = F.relu(self.bn1(x))
#         # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         # out = self.conv1(out)
#         # out = self.conv2(F.relu(self.bn2(out)))
#         # return out + shortcut
#         # return shortcut


class ResNet18(BaseNet):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__(loss_fn=F.cross_entropy)
        
    #     # self.in_channels = 64
        
    #     # self.prep = nn.Sequential(
    #     #     nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False),
    #     #     # HalfBatchNorm2d(64), # !! Removed per David Page
    #     #     # nn.ReLU()
    #     # )
        
    #     # # self.layers = nn.Sequential(
    #     #     # self._make_layer(64, 512, num_blocks[0], stride=1),
    #     #     # self._make_layer(64, 128, num_blocks[1], stride=2),
    #     #     # self._make_layer(128, 256, num_blocks[2], stride=2),
    #     #     # self._make_layer(256, 512, num_blocks[3], stride=2), # !! Edited per David Page
    #     # # )
        
        self.classifier = nn.Linear(512, num_classes)
        
    # def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        
    #     strides = [stride] + [1] * (num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
    #         in_channels = out_channels
        
    #     return nn.Sequential(*layers)
    
    def forward(self, x):
        # x = x.half()
        # x = self.prep(x)
        # # x = self.layers(x)
        # # !! Removed avg pool per David Page
        # x = F.adaptive_max_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

# --
# Define model

print('cifar10.py: initializing model...', file=sys.stderr)

cuda = torch.device('cuda')
model = ResNet18()#.to(cuda).half()
model.verbose = False
print(model, file=sys.stderr)

# --
# Initialize optimizer

print('cifar10.py: initializing optimizer...', file=sys.stderr)

lr_scheduler = HPSchedule.one_cycle(hp_max=args.lr_max, epochs=args.epochs, extra=0)

model.init_optimizer(
    opt=torch.optim.SGD,
    params=model.parameters(),
    hp_scheduler={"lr" : lr_scheduler},
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    nesterov=True,
)

# for v in model.children(): 
#     v.half()

# --
# Train

print('cifar10.py: training...', file=sys.stderr)
t = time()
for epoch in range(args.epochs):
    # train = model.train_epoch(dataloaders, mode='train')#, metric_fns=['n_correct'])
    test  = model.predict(dataloaders, mode='train')#, metric_fns=['n_correct'])
    print(json.dumps({
        "epoch"     : int(epoch),
        # "lr"        : model.hp['lr'],
        # "test_acc"  : float(test['acc']),
        # "train_acc" : float(train['acc']),
        "time"      : time() - t,
    }))
    sys.stdout.flush()

# model.save('page_weights')

# # --
# # Output late features

# class Eye(nn.Module):
#     def forward(self, x):
#         return x

# model.classifier = Eye()

# train_feats, train_targets = model.predict(dataloaders, mode='train')
# test_feats, test_targets   = model.predict(dataloaders, mode='test')

# torch.save(train_feats,   './feats/page_train_feats')
# torch.save(train_targets, './feats/page_train_targets')
# torch.save(test_feats,    './feats/page_test_feats')
# torch.save(test_targets,  './feats/page_test_targets')
