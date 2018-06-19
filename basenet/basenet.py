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
    # !! Do we want to always turn off `training` mode when the layer is frozen?
    x.training = False if getattr(x, 'frozen', False) else mode
    for module in x.children():
        _set_train(module, mode)
    
    return x

def _to_device(x, device):
    if TORCH_VERSION_4:
        if isinstance(x, tuple) or isinstance(x, list):
            return [xx.to(device) for xx in x]
        else:
            return x.to(device)
    else:
        if device == 'cuda':
            return x.cuda()
        elif device == 'cpu':
            return x.cpu()
        else:
            raise Exception

class Metrics:
    @staticmethod
    def n_correct(output, target):
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]
        if isinstance(target, tuple) or isinstance(output, list):
            target = target[0]
        
        correct = (output.max(dim=-1)[1] == target).long().sum()
        return int(correct)

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
        if TORCH_VERSION_4:
            super().to(device=device)
        else:
            if device == 'cuda':
                self.cuda()
            elif device == 'cpu':
                self.cpu()
            else:
                raise Exception
        
        return self
    
    # --
    # Optimization
    
    def _filter_requires_grad(self, params):
        # User shouldn't be passing variables that don't require gradients
        if isinstance(params[0], dict):
            check = np.all([np.all([pp.requires_grad for pp in p['params']]) for p in params]) 
        else:
            check = np.all([p.requires_grad for p in params])
        
        if not check:
            warnings.warn((
                'BaseNet.init_optimizer: some variables do not require gradients. '
                'Ignoring them, but better to handle explicitly'
            ), RuntimeWarning)
        
        return params
    
    def init_optimizer(self, opt, params, hp_scheduler=None, clip_grad_norm=0, **kwargs):
        params = list(params)
        
        self.clip_grad_norm = clip_grad_norm
        self.hp_scheduler = hp_scheduler
        
        if hp_scheduler is not None:
            for hp_name, scheduler in hp_scheduler.items():
                assert hp_name not in kwargs.keys(), '%s in kwargs.keys()' % hp_name
                kwargs[hp_name] = scheduler(0)
        
        params = self._filter_requires_grad(params)
        
        self.opt = opt(params, **kwargs)
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
    
    def train_batch(self, data, target, metric_fns=None):
        assert self.loss_fn is not None, 'BaseNet: self.loss_fn is None'
        assert self.training, 'BaseNet: self.training == False'
        
        self.opt.zero_grad()
        
        if not TORCH_VERSION_4:
            data, target = Variable(data), Variable(target)
        
        data, target = _to_device(data, self.device), _to_device(target, self.device)
        
        output = self(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        
        if self.clip_grad_norm > 0:
            if TORCH_VERSION_4:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad_norm)
        
        self.opt.step()
        
        metrics = [m(output, target) for m in metric_fns] if metric_fns is not None else []
        return float(loss), metrics
    
    def eval_batch(self, data, target, metric_fns=None):
        assert not self.training, 'BaseNet: self.training == True'
        
        def _eval(data, target, metric_fns):
            data, target = _to_device(data, self.device), _to_device(target, self.device)
            
            output = self(data)
            loss = self.loss_fn(output, target)
            
            metrics = [m(output, target) for m in metric_fns] if metric_fns is not None else []
            return float(loss), metrics
        
        if TORCH_VERSION_4:
            with torch.no_grad():
                return _eval(data, target, metric_fns)
        else:
            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            return _eval(data, target, metric_fns)
    
    # --
    # Epoch steps
    
    def _run_epoch(self, dataloaders, mode, batch_fn, set_progress, desc, num_batches=np.inf, compute_acc=False):
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='%s:%s' % (desc, mode))
            
            metric_fns = []
            if compute_acc:
                metric_fns = [Metrics.n_correct]
            
            if hasattr(self, 'reset'):
                self.reset()
            
            correct, total, loss_hist = 0, 0, [None] * len(loader)
            for batch_idx, (data, target) in gen:
                if set_progress:
                    self.set_progress(self.epoch + batch_idx / len(loader))
                
                loss, metrics = batch_fn(data, target, metric_fns=metric_fns)
                
                loss_hist[batch_idx] = loss
                if compute_acc:
                    correct += metrics[0]
                    total   += target.shape[0]
                
                if batch_idx > num_batches:
                    break
                
                if self.verbose:
                    gen.set_postfix(**{
                        "acc"  : correct / total if compute_acc else -1.0,
                        "loss" : loss,
                    })
            
            if self.verbose:
                gen.set_postfix(**{
                    "acc"          : correct / total if compute_acc else -1.0,
                    "last_10_loss" : np.mean(loss[-10:]),
                })
            
            if set_progress:
                self.epoch += 1
            
            return {
                "acc"  : float(correct / total) if compute_acc else -1.0,
                "loss" : list(map(float, loss_hist)),
            }
    
    def train_epoch(self, dataloaders, mode='train', **kwargs):
        assert self.opt is not None, "BaseNet: self.opt is None"
        _ = self.train()
        return self._run_epoch(
            dataloaders=dataloaders,
            mode=mode,
            batch_fn=self.train_batch,
            set_progress=True,
            desc="train_epoch",
            **kwargs,
        )
        
    def eval_epoch(self, dataloaders, mode='val', **kwargs):
        _ = self.eval()
        return self._run_epoch(
            dataloaders=dataloaders,
            mode=mode,
            batch_fn=self.eval_batch,
            set_progress=False,
            desc="eval_epoch",
            **kwargs,
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
            
            if hasattr(self, 'reset'):
                self.reset()
            
            for _, (data, target) in gen:
                if TORCH_VERSION_4:
                    with torch.no_grad():
                        output = self(_to_device(data, self.device)).cpu()
                else:
                    data = Variable(data, volatile=True)
                    output = self(_to_device(data, self.device)).cpu()
                
                all_output.append(output)
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

