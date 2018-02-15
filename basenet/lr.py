#!/usr/bin/env python

"""
    lr.py
    
    learning rate scheduler
"""

from __future__ import print_function, division

import sys
import numpy as np

# --
# Helpers

def power_sum(base, k):
    return (base ** (k + 1) - 1) / (base - 1)


def inv_power_sum(x, base):
    return np.log(x * (base - 1) + 1) / np.log(base) - 1

# --

class LRSchedule(object):
    
    @staticmethod
    def set_lr(optimizer, lr):
        num_param_groups = len(list(optimizer.param_groups))
        
        if isinstance(lr, float):
            lr = [lr] * num_param_groups
        else:
            assert len(lr) == num_param_groups, "len(lr) != num_param_groups"
        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[i]
    
    @staticmethod
    def constant(lr_init=0.1, **kwargs):
        def f(progress):
            return lr_init
        
        return f
    
    @staticmethod
    def step(lr_init=0.1, breaks=(150, 250), factors=(0.1, 0.1), **kwargs):
        """ Step function learning rate annealing """
        assert len(breaks) == len(factors)
        breaks = np.array(breaks)
        def f(progress):
            return lr_init * np.prod(factors[:((progress >= breaks).sum())])
        
        return f
    
    @staticmethod
    def linear(lr_init=0.1, epochs=10, **kwargs):
        def f(progress):
            """ Linear learning rate annealing """
            return lr_init * float(epochs - progress) / epochs
        
        return f
    
    @staticmethod
    def cyclical(lr_init=0.1, lr_burn_in=0.05, epochs=10, **kwargs):
        def f(progress):
            """ Cyclical learning rate w/ annealing """
            return lr_init * (1 - progress % 1) * (epochs - np.floor(progress)) / epochs
        
        return f
    
    @staticmethod
    def sgdr(lr_init=0.1, period_length=50, lr_min=0, t_mult=1, **kwargs):
        print('sgdr: period_length=%d | lr_init=%s' % (period_length, str(lr_init)), file=sys.stderr)
        def f(progress):
            """ SGDR learning rate annealing """
            if t_mult > 1:
                period_id = np.floor(inv_power_sum(progress / period_length, t_mult)) + 1
                offsets = power_sum(t_mult, period_id - 1) * period_length
                period_progress = (progress - offsets) / (t_mult ** period_id * period_length)
            
            else:
                period_progress = (progress % period_length) / period_length
            
            return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(period_progress * np.pi))
        
        return f
    
    @staticmethod
    def exponential_increase(lr_init=0.1, lr_max=10, num_steps=100, **kwargs):
        mult = (lr_max / lr_init) ** (1 / num_steps)
        def f(progress):
            return lr_init * mult ** progress
            
        return f

if __name__ == "__main__":
    from rsub import *
    from matplotlib import pyplot as plt
    
    # # Step
    # lr = LRSchedule.step(lr_init=np.array([1, 2]), factors=(0.5, 0.5), breaks=(10, 20))
    # lrs = np.vstack([lr(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(lrs[:,0])
    # _ = plt.plot(lrs[:,1])
    # show_plot()
    
    # # Linear
    # lr = LRSchedule.linear(epochs=30, lr_init=np.array([1, 2]))
    # lrs = np.vstack([lr(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(lrs[:,0])
    # _ = plt.plot(lrs[:,1])
    # show_plot()
    
    # # Cyclical
    # lr = LRSchedule.cyclical(epochs=30, lr_init=np.array([1, 2]))
    # lrs = np.vstack([lr(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(lrs[:,0])
    # _ = plt.plot(lrs[:,1])
    # show_plot()
    
    # # SGDR
    # lr = LRSchedule.sgdr(period_length=10, t_mult=2, lr_init=np.array([1, 2]))
    # lrs = np.vstack([lr(i) for i in np.linspace(0, 30, 1000)])
    # _ = plt.plot(lrs[:,0])
    # _ = plt.plot(lrs[:,1])
    # show_plot()
    
    # exponential increase (for setting learning rates)
    lr = LRSchedule.exponential_increase(lr_init=np.array([1e-5, 1e-4]), lr_max=10, num_steps=100)
    lrs = np.vstack([lr(i) for i in np.linspace(0, 100, 1000)])
    _ = plt.plot(lrs[:,0])
    _ = plt.plot(lrs[:,1])
    _ = plt.yscale('log')
    show_plot()
