#!/usr/bin/env python

import os
import sys
import json
import pandas as pd
import numpy as np
from rsub import *
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import pylab as pl

def smart_json_loads(x):
    try:
        return json.loads(x)
    except:
        pass

colors = pl.cm.jet(np.linspace(0,1,101))

all_data = []
for p in sys.argv[1:]:
    data = list(filter(None, map(smart_json_loads, open(p))))
    
    acc   = [d['test_acc'] for d in data]
    epoch = [d['epoch'] for d in data]
    _ = plt.plot(acc, alpha=0.75, label=p)


_ = plt.legend(loc='lower right')
_ = plt.grid(alpha=0.25)
for t in np.arange(0.90, 1.0, 0.01):
    _ = plt.axhline(t, c='grey', alpha=0.25, lw=1)

_ = plt.ylim(0.5, 1.0)
# _ = plt.xlim(0, 40)
show_plot()
