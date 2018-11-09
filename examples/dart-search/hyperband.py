"""
    hyperband.py
    
    `hyperband` algorithm for hyper-parameter optimization
    
    Copied/adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
    
    The algorithm is motivated by the idea that random search is pretty 
    good as far as black-box optimization goes, so let's try to do it faster.
"""

from __future__ import print_function, division

import sys
import numpy as np
import ujson as json
from math import log, ceil

class HyperBand:
    
    def __init__(self, model, max_iter=81, eta=3):
        
        self.model = model
        
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(log(max_iter) // log(eta))
        self.B = (self.s_max + 1) * max_iter
        
        self.best_obj = np.inf
        self.history = []
    
    def run(self):
        for s in reversed(range(self.s_max + 1)):
            
            # initial number of configs
            n = int(ceil(self.B // self.max_iter // (s + 1) * self.eta ** s)) 
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s) 
            
            # initial configs
            configs = self.model.random_configs(s=s, n=n)
            
            for i in range(s + 1):
                
                # number of iterations for these configs
                r_i = r * self.eta ** i
                
                print("\n -------------------- %d configs @ %d iterations -------------------- \n" % (len(configs), int(round(r_i))), file=sys.stderr)
                
                results = []
                for j, config in enumerate(configs):
                    print("Config %d: %s" % (j, json.dumps(config)), file=sys.stderr)
                    
                    res = self.model.eval_config(config=config, iters=r_i)
                    results.append(res)
                    
                    self.best_obj = min(res['obj'], self.best_obj)
                    print("Current: %f | Best: %f" % (float(res['obj']), self.best_obj), file=sys.stderr)
                    print(json.dumps(res), file=sys.stderr)
                
                self.history += results
                
                # Sort by objective value
                results = sorted(results, key=lambda x: x['obj'])
                
                # Drop models that have already converged
                results = list(filter(lambda x: not x.get('converged', False), results))
                
                # Determine how many configs to keep
                n_keep = int(n * self.eta ** (-i - 1))
                configs = [result['config'] for result in results[:n_keep]]

