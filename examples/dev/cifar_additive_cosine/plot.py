from rsub import *
from matplotlib import pyplot as plt

import json
import pandas as pd
from glob import glob

data = []
for f in glob('./results/seed_123/*'):
    x = list(map(json.loads, open(f)))
    for xx in x:
        xx.update({'run' : f})
    
    data += x

df = pd.DataFrame(data)

z = df.sort_values('test_acc', ascending=False).drop_duplicates('run')
z = z[z.test_acc > 0.80]

_ = plt.scatter(z.ac_s, z.ac_m, c=z.test_acc)
_ = plt.colorbar()
show_plot()

_ = plt.scatter(z.ac_m, z.test_acc, c=z.ac_s)
show_plot()