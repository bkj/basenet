#!/usr/bin/env python

"""
    basenet.py
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .helpers import to_numpy
from .lr import LRSchedule

class BaseNet(nn.Module):
    
    def __init__(self, loss_fn=F.cross_entropy):
        super(BaseWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.opt = None
        self.progress = 0
        self.epoch = 0
        self.lr = -1
    
    # --
    # Optimization
    
    def init_optimizer(self, opt, params, lr_scheduler, **kwargs):
        assert 'lr' not in kwargs, "BaseWrapper.init_optimizer: can't set LR from outside"
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
            self.opt = opt(params, lr=self.lr_scheduler(0), **kwargs)
    
    def set_progress(self, progress):
        self.progress, self.lr = progress, self.lr_scheduler(progress)
        LRSchedule.set_lr(self.opt, self.lr)
    
    # --
    # Batch steps
    
    def train_batch(self, data, target):
        _ = self.train()
        self.opt.zero_grad()
        output = self(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.opt.step()
        return output
    
    def eval_batch(self, data, target):
        _ = self.eval()
        output = self(data)
        return (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
    
    # --
    # Epoch steps
    
    def train_epoch(self, dataloaders, num_batches=np.inf):
        assert self.opt is not None, "BaseWrapper: self.opt is None"
        
        loader = dataloaders['train']
        correct, total = 0, 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            
            self.set_progress(self.epoch + batch_idx / len(loader))
            
            output = self.train_batch(data, target)
            
            correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
            total += data.shape[0]
            
            if batch_idx > num_batches:
                break
        
        self.epoch += 1
        return correct / total
    
    def eval_epoch(self, dataloaders, mode='val', num_batches=np.inf):
        assert self.opt is not None, "BaseWrapper: self.opt is None"
        
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            _ = self.eval()
            correct, total = 0, 0 
            for batch_idx, (data, target) in enumerate(loader):
                data = Variable(data.cuda(), volatile=True)
                
                output = self(data)
                
                correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
                total += data.shape[0]
                
                if batch_idx > num_batches:
                    break
            
            return correct / total


class BaseWrapper(BaseNet):
    def __init__(self, net=None, **kwargs):
        super(BaseWrapper, self).__init__(**kwargs)
        self.net = net
    
    def forward(self, x):
        return self.net(x)

