#!/usr/bin/env python

import os
import sys
import json
import pandas as pd
import numpy as np

from rsub import *
from matplotlib import pyplot as plt

def smart_json_loads(x):
    try:
        return json.loads(x)
    except:
        pass

all_data = []
for p in sys.argv[1:]:
    data = list(filter(None, map(smart_json_loads, open(p))))
    acc   = [1 - d['test_err'] for d in data]
    epoch = [d['epoch'] for d in data]
    _ = plt.scatter(epoch, acc, label=os.path.basename(p), s=3, alpha=0.25)

_ = plt.grid(alpha=0.25)
for t in np.arange(0.90, 1.0, 0.01):
    _ = plt.axhline(t, c='grey', alpha=0.25, lw=1)

_ = plt.legend(loc='lower right')
_ = plt.ylim(0.5, 1.0)
show_plot()
