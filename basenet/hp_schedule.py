#!/usr/bin/env python

"""
    hp_schedule.py
    
    Optimizer hyperparameter scheduler
"""

from __future__ import print_function, division

import sys
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# --
# Helpers

def power_sum(base, k):
    return (base ** (k + 1) - 1) / (base - 1)


def inv_power_sum(x, base):
    return np.log(x * (base - 1) + 1) / np.log(base) - 1


def linterp(x, start_x, end_x, start_y, end_y):
    return start_y + (x - start_x) / (end_x - start_x) * (end_y - start_y)

def _set_hp(optimizer, hp_name, hp_hp):
    if isinstance(hp_hp, float):
        hp_hp = [hp_hp] * num_param_groups
    else:
        assert len(hp_hp) == num_param_groups, ("len(%s) != num_param_groups" % hp_name)
    
    for i, param_group in enumerate(optimizer.param_groups):
        param_group[hp_name] = hp_hp[i]

# --

class HPSchedule(object):
    
    @staticmethod
    def set_hp(optimizer, hp):
        num_param_groups = len(list(optimizer.param_groups))
        
        for hp_name, hp_hp in hp.items():
            _set_hp(optimizer, hp_name, hp_hp)
    
    @staticmethod
    def constant(hp_init=0.1, **kwargs):
        def f(progress):
            return hp_init
        
        return f
    
    @staticmethod
    def step(hp_init=0.1, breaks=(150, 250), factors=(0.1, 0.1), **kwargs):
        """ Step function learning rate annealing """
        assert len(breaks) == len(factors)
        breaks = np.array(breaks)
        def f(progress):
            return hp_init * np.prod(factors[:((progress >= breaks).sum())])
        
        return f
    
    @staticmethod
    def linear(hp_init=0.1, epochs=10, **kwargs):
        def f(progress):
            """ Linear learning rate annealing """
            return hp_init * float(epochs - progress) / epochs
        
        return f
    
    @staticmethod
    def linear_cycle(hp_init=0.1, epochs=10, low_hp=0.005, extra=5, **kwargs):
        def f(progress):
            if progress < epochs / 2:
                return 2 * hp_init * (1 - float(epochs - progress) / epochs)
            elif progress <= epochs:
                return low_hp + 2 * hp_init * float(epochs - progress) / epochs
            elif progress <= epochs + extra:
                return low_hp * float(extra - (progress - epochs)) / extra
            else:
                return low_hp / 10
        
        return f
    
    @staticmethod
    def piecewise_linear(breaks, hps, **kwargs):
        assert len(breaks) == len(hps)
        
        def _f(progress):
            if progress < breaks[0]:
                return hps[0]
            
            for i in range(1, len(breaks)):
                if progress < breaks[i]:
                    return linterp(progress, breaks[i - 1], breaks[i], hps[i - 1], hps[i])
            
            return hps[-1]
        
        def f(x):
            if isinstance(x, list) or isinstance(x, np.ndarray):
                return [_f(xx) for xx in x]
            else:
                return _f(x)
        
        return f
    
    @staticmethod
    def cyclical(hp_init=0.1, hp_burn_in=0.05, epochs=10, **kwargs):
        def f(progress):
            """ Cyclical learning rate w/ annealing """
            return hp_init * (1 - progress % 1) * (epochs - np.floor(progress)) / epochs
        
        return f
    
    @staticmethod
    def sgdr(hp_init=0.1, period_length=50, hp_min=0, t_mult=1, **kwargs):
        def f(progress):
            """ SGDR learning rate annealing """
            if t_mult > 1:
                period_id = np.floor(inv_power_sum(progress / period_length, t_mult)) + 1
                offsets = power_sum(t_mult, period_id - 1) * period_length
                period_progress = (progress - offsets) / (t_mult ** period_id * period_length)
            
            else:
                period_progress = (progress % period_length) / period_length
            
            return hp_min + 0.5 * (hp_init - hp_min) * (1 + np.cos(period_progress * np.pi))
        
        return f
    
    @staticmethod
    def burnin_sgdr(hp_init=0.1, burnin_progress=0.15, burnin_factor=100, **kwargs):
        sgdr = HPSchedule.sgdr(hp_init=hp_init, **kwargs)
        
        def f(progress):
            """ SGDR learning rate annealing, w/ constant burnin period """
            if progress < burnin_progress:
                return hp_init / burnin_factor
            else:
                return sgdr(progress)
        
        return f
    
    @staticmethod
    def exponential_increase(hp_init=0.1, hp_max=10, num_steps=100, **kwargs):
        mult = (hp_max / hp_init) ** (1 / num_steps)
        def f(progress):
            return hp_init * mult ** progress
            
        return f

# --
# HP Finder

class HPFind(object):
    
    @staticmethod
    def find(model, dataloaders, hp_init=1e-5, hp_max=10, hp_mults=None, params=None, mode='train', smooth_loss=False):
        assert mode in dataloaders, '%s not in loader' % mode
        
        # --
        # Setup HP schedule
        
        if model.verbose:
            print('HPFind.find: copying model')
        
        model = copy.deepcopy(model)
        
        if hp_mults is not None:
            hp_init *= hp_mults
            hp_max *= hp_mults # Correct?
        
        hp_scheduler = HPSchedule.exponential_increase(hp_init=hp_init, hp_max=hp_max, num_steps=len(dataloaders[mode]))
        
        if params is None:
            params = model.parameters()
        
        model.init_optimizer(
            opt=torch.optim.SGD,
            params=params,
            hp_scheduler=hp_scheduler,
            momentum=0.9
        )
        
        # --
        # Run epoch of training w/ increasing learning rate
        
        avg_mom  = 0.98 # For smooth_loss
        avg_loss = 0.   # For smooth_loss
        
        hp_hist, loss_hist = [], []
        
        gen = enumerate(dataloaders[mode])
        if model.verbose:
            gen = tqdm(gen, total=len(dataloaders[mode]), desc='HPFind.find:')
        
        for batch_idx, (data, target) in gen:
            
            model.set_progress(batch_idx)
            
            _, loss = model.train_batch(data, target)
            if smooth_loss:
                avg_loss    = avg_loss * avg_mom + loss * (1 - avg_mom)
                debias_loss = avg_loss / (1 - avg_mom ** (batch_idx + 1))
                loss_hist.append(debias_loss)
            else:
                loss_hist.append(loss)
            
            hp_hist.append(model.hp)
            
            if loss > np.min(loss_hist) * 4:
                break
        
        return np.vstack(hp_hist), loss_hist
    
    @staticmethod
    def get_optimal_hp(hp_hist, loss_hist, c=10, burnin=5):
        """
            For now, gets smallest loss and goes back an order of magnitude
            Maybe it'd be better to use the point w/ max slope?  Or not use smoothed estimate? 
        """
        hp_hist, loss_hist = hp_hist[burnin:], loss_hist[burnin:]
        
        min_loss_idx = np.array(loss_hist).argmin()
        min_loss_hp = hp_hist[min_loss_idx]
        opt_hp = min_loss_hp / c
        
        if len(opt_hp) == 1:
            opt_hp = opt_hp[0]
        
        return opt_hp


if __name__ == "__main__":
    from rsub import *
    from matplotlib import pyplot as plt
    
    # # Step
    # hp = HPSchedule.step(hp_init=np.array([1, 2]), factors=(0.5, 0.5), breaks=(10, 20))
    # hps = np.vstack([hp(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # show_plot()
    
    # # Linear
    # hp = HPSchedule.linear(epochs=30, hp_init=np.array([1, 2]))
    # hps = np.vstack([hp(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # show_plot()
    
    # Linear cycle
    # hp = HPSchedule.linear_cycle(epochs=30, hp_init=0.1, extra=10)
    # hps = np.vstack([hp(i) for i in np.linspace(0, 40, 1000)])
    # _ = plt.plot(hps)
    # show_plot()
    
    # Piecewise linear
    hp = HPSchedule.piecewise_linear(breaks=[0, 5, 10, 15], hps=[0, 1, 0.25, 0])
    hps = np.vstack([hp(i) for i in np.linspace(-1, 16, 1000)])
    _ = plt.plot(hps)
    show_plot()
    
    # # Cyclical
    # hp = HPSchedule.cyclical(epochs=30, hp_init=np.array([1, 2]))
    # hps = np.vstack([hp(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # show_plot()
    
    # # SGDR
    # hp = HPSchedule.sgdr(period_length=10, t_mult=2, hp_init=np.array([1, 2]))
    # hps = np.vstack([hp(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # show_plot()
    
    # exponential increase (for setting learning rates)
    # hp = HPSchedule.exponential_increase(hp_init=np.array([1e-5, 1e-4]), hp_max=10, num_steps=100)
    # hps = np.vstack([hp(i) for i in np.linspace(0, 100, 1000)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # _ = plt.yscale('log')
    # show_plot()