#!/usr/bin/env python

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
    
    acc   = [d['test_acc'] for d in data]
    epoch = [d['epoch'] for d in data]
    _ = plt.plot(acc, alpha=0.75, label=p, c='red')
    _ = plt.plot(np.array(pd.Series(acc).cummax()), alpha=0.75, c='blue')


_ = plt.legend(loc='lower right')
_ = plt.axhline(0.9, c='grey', alpha=0.5)
# _ = plt.axhline(0.94, c='grey', alpha=0.5)
_ = plt.ylim(0.5, 1.0)
# _ = plt.xlim(0, 40)
show_plot()
