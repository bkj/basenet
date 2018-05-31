#!/usr/bin/env python

"""
    basenet.py
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from tqdm import tqdm
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .helpers import to_numpy
from .hp_schedule import HPSchedule

TORCH_VERSION_4 = '0.4' == torch.__version__[:3]

# --
# Helpers

def _set_train(x, mode):
    x.training = False if getattr(x, 'frozen', False) else mode
    for module in x.children():
        _set_train(module, mode)
    
    return x

def _to_device(x, device):
    if isinstance(x, tuple) or isinstance(x, list):
        return [xx.to(device) for xx in x]
    else:
        return x.to(device)

# --
# Model

class BaseNet(nn.Module):
    
    def __init__(self, loss_fn=F.cross_entropy, verbose=False):
        super().__init__()
        self.loss_fn = loss_fn
        
        self.opt          = None
        self.hp_scheduler = None
        self.hp           = None
        
        self.progress = 0
        self.epoch    = 0
        
        self.verbose = verbose
        self.device = None
    
    def to(self, device=None):
        self.device = device
        super().to(device=device)
        return self
    
    # --
    # Optimization
    
    def init_optimizer(self, opt, params, hp_scheduler=None, clip_grad_norm=0, **kwargs):
        params = list(params)
        
        self.clip_grad_norm = clip_grad_norm
        self.hp_scheduler = hp_scheduler
        
        if hp_scheduler is not None:
            for hp_name, scheduler in hp_scheduler.items():
                kwargs[hp_name] = scheduler(0)
        
        if not np.all([p.requires_grad for p in params]):
            warnings.warn((
                'BaseNet.init_optimizer: some variables do not require gradients. '
                'Ignoring them, but better to handle explicitly'
            ), RuntimeWarning)
        
        self.opt = opt([p for p in params if p.requires_grad], **kwargs)
        self.set_progress(0)
    
    def set_progress(self, progress):
        self.progress = progress
        self.epoch = np.floor(progress)
        
        if self.hp_scheduler is not None:
            self.hp = dict([(hp_name, scheduler(progress)) for hp_name,scheduler in self.hp_scheduler.items()])
            HPSchedule.set_hp(self.opt, self.hp)
    
    # --
    # Training states
    
    def train(self, mode=True):
        """ have to override this function to allow more finegrained control """
        return _set_train(self, mode=mode)
    
    # --
    # Batch steps
    
    def train_batch(self, data, target):
        _ = self.train()
        
        data, target = _to_device(data, self.device), _to_device(target, self.device)
        
        self.opt.zero_grad()
        output = self(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad_norm)
        
        self.opt.step()
        
        return output, float(loss)
    
    
    def eval_batch(self, data, target):
        _ = self.eval()
        
        with torch.no_grad():
            data, target = _to_device(data, self.device), _to_device(target, self.device)
            
            output = self(data)
            loss = self.loss_fn(output, target)
            
            return output, float(loss)
    
    # --
    # Epoch steps
    
    def _run_epoch(self, dataloaders, mode, num_batches, batch_fn, set_progress, desc):
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='%s:%s' % (desc, mode))
            
            
            correct, total, loss_hist = 0, 0, [None] * len(loader)
            for batch_idx, (data, target) in gen:
                if set_progress:
                    self.set_progress(self.epoch + batch_idx / len(loader))
                
                output, loss = batch_fn(data, target)
                loss_hist[batch_idx] = loss
                
                # >>
                # !! Ugly hack for experiments
                
                if isinstance(target, list):
                    correct += (to_numpy(output).argmax(axis=1) == to_numpy(target[0])).sum()
                else:
                    correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
                
                if isinstance(data, list):
                    total += data[0].shape[0]
                else:
                    total += data.shape[0]
                
                # <<
                
                if batch_idx > num_batches:
                    break
                
                if self.verbose:
                    gen.set_postfix(acc=correct / total)
            
            if set_progress:
                self.epoch += 1
            
            return {
                "acc"  : float(correct / total),
                "loss" : list(map(float, loss_hist)),
            }
    
    def train_epoch(self, dataloaders, mode='train', num_batches=np.inf):
        assert self.opt is not None, "BaseWrapper: self.opt is None"
        
        return self._run_epoch(
            dataloaders=dataloaders,
            mode=mode,
            num_batches=num_batches,
            
            batch_fn=self.train_batch,
            set_progress=True,
            desc="train_epoch",
        )
        
    def eval_epoch(self, dataloaders, mode='val', num_batches=np.inf):
        
        return self._run_epoch(
            dataloaders=dataloaders,
            mode=mode,
            num_batches=num_batches,
            
            batch_fn=self.eval_batch,
            set_progress=False,
            desc="eval_epoch",
        )
    
    def predict(self, dataloaders, mode='val'):
        _ = self.eval()
        
        all_output, all_target = [], []
        
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='predict:%s' % mode)
            
            for _, (data, target) in gen:
                with torch.no_grad():
                    data = _to_device(data, self.device)
                    all_output.append(self(data).cpu())
                    all_target.append(target)
        
        return torch.cat(all_output), torch.cat(all_target)
    
    def save(self, outpath):
        torch.save(self.state_dict(), outpath)
    
    def load(self, inpath):
        self.load_state_dict(torch.load(inpath))


class BaseWrapper(BaseNet):
    def __init__(self, net=None, **kwargs):
        super().__init__(**kwargs)
        self.net = net
    
    def forward(self, x):
        return self.net(x)

