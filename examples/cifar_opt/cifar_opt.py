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
from datetime import datetime
from collections import OrderedDict

from basenet import BaseNet
from basenet.hp_schedule import HPSchedule
from basenet.helpers import to_numpy, set_seeds

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

from torchvision import transforms, datasets

import dlib

# --
# Helpers

def dlib_find_max_global(f, bounds, **kwargs):
    varnames = f.__code__.co_varnames[:f.__code__.co_argcount]
    bound1_, bound2_ = [], []
    for varname in varnames:
        bound1_.append(bounds[varname][0])
        bound2_.append(bounds[varname][1])
    
    return dlib.find_max_global(f, bound1_, bound2_, **kwargs)


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=789)
    parser.add_argument('--download', action="store_true")
    return parser.parse_args()

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
        x = self.prep(x)#.half())
        
        x = self.layers(x)
        
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)
        
        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)
        
        x = torch.cat([x_avg, x_max], dim=-1)
        
        x = self.classifier(x)
        
        return x


if __name__ == "__main__":
    args = parse_args()
    
    set_seeds(args.seed)
    
    # --
    # IO
    
    cifar10_stats = {
        "mean" : (0.4914, 0.4822, 0.4465),
        "std"  : (0.24705882352941178, 0.24352941176470588, 0.2615686274509804),
    }
    
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: np.asarray(x)),
        transforms.Lambda(lambda x: np.pad(x, [(4, 4), (4, 4), (0, 0)], mode='reflect')),
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.RandomCrop(32),
        
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
    
    def run_one(break1, break2, val1, val2):
        
        # try:
            # set_seeds(args.seed) # Might have bad side effects
            
            if (break1 >= break2):
                return float(-1)
            
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            params = OrderedDict([
                ("timestamp",    timestamp),
                ("break1",       break1),
                ("break2",       break2),
                ("val1",         10 ** val1),
                ("val2",         10 ** val2),
                ("momentum",     args.momentum),
                ("weight_decay", args.weight_decay),
            ])
            
            model = ResNet18().cuda()#.half()
            
            lr_scheduler = HPSchedule.piecewise_linear(
                breaks=[0, break1, break2, args.epochs],
                vals=[0, 10 ** val1, 10 ** val2, 0]
            )
            
            model.init_optimizer(
                opt=torch.optim.SGD,
                params=model.parameters(),
                hp_scheduler={"lr" : lr_scheduler},
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True,
            )
            
            t = time()
            for epoch in range(args.epochs):
                train = model.train_epoch(dataloaders, mode='train')
                test  = model.eval_epoch(dataloaders, mode='test')
                
                res = OrderedDict([
                    ("params",   params),
                    ("epoch",    int(epoch)),
                    ("lr",       model.hp['lr']),
                    ("test_acc", float(test['acc'])),
                    ("time",     time() - t),
                ])
                print(json.dumps(res))
                sys.stdout.flush()
            
            return float(test['acc'])
        # except:
            # return float(-1)
    
    print('cifar_opt.py: start', file=sys.stderr)
    best_args, best_score = dlib_find_max_global(run_one, bounds={
        "break1" : (0, args.epochs),
        "break2" : (0, args.epochs),
        "val1"   : (-3, 0),
        "val2"   : (-3, 0),
    }, num_function_calls=100, solver_epsilon=0.001)
    
    print(best_args, file=sys.stderr)
    print(best_score, file=sys.stderr)
    print('cifar_opt.py: done', file=sys.stderr)
