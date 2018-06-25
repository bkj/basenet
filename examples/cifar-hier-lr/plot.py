#!/usr/bin/env python

"""
    plot.py
"""

from rsub import *
from matplotlib import pyplot as plt

import os
import json
import pandas as pd
from glob import glob

pd.set_option('display.width', 120)

dfs = []
for f in sorted(glob('results/*')):
    x = [json.loads(xx) for xx in open(f)]
    
    df = pd.DataFrame(x)
    df['path'] = os.path.basename(f)
    dfs.append(df)

df = pd.concat(dfs)
df['macro'] = df.path.apply(lambda x: x.split('-')[0])
df['micro'] = df.path.apply(lambda x: x.split('-')[1].split('_')[0])
df['lr']    = df.path.apply(lambda x: float(x.split('_')[1].split('.')[0]))

# >>
df = df[df.lr == 0.2]
# <<

clookup = {
    "step"     : "orange",
    "constant" : "green",
    "linear"   : "blue",
    "cosine"   : "red",
}

_ = df.groupby(['micro', 'macro']).apply(
    lambda x: plt.plot(x.test_acc, label=x.path.iloc[0], c=clookup[x.macro.iloc[0]], alpha=0.5))
_ = plt.legend(fontsize=8)
# _ = plt.ylim(0.9, 0.95)
_ = plt.axhline(0.93)
show_plot()

_ = df.groupby(['micro', 'macro']).apply(
    lambda x: plt.plot(x.test_acc, label=x.path.iloc[0], c=clookup[x.micro.iloc[0]], alpha=0.5))
_ = plt.legend(fontsize=8)
# _ = plt.ylim(0.9, 0.95)
_ = plt.axhline(0.93)
show_plot()