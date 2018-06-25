"""

Macro
    Constant
    Step
    Linear
    Cosine
    # Piecewise-linear -- skip for now

Micro
    Constant
    Step -- too many parameters
    Linear
    Cosine
    # Piecewise-linear -- too many parameters

"""

import numpy as np
from basenet import HPSchedule

from rsub import *
from matplotlib import pyplot as plt

# --
# Define schedules

hp_max       = 0.2
macro_epochs = 30
micro_epochs = 1

macro = {
    "constant" : HPSchedule.constant(hp_max=hp_max),
    "step"     : HPSchedule.step(hp_max=hp_max, epochs=macro_epochs, breaks=(10, 20), factors=(0.1, 0.1)),
    "linear"   : HPSchedule.linear(hp_max=hp_max, epochs=macro_epochs),
    "cosine"   : HPSchedule.sgdr(hp_max=hp_max, period_length=macro_epochs),
}

micro = {
    "constant" : HPSchedule.constant(hp_max=1),
    "step"     : HPSchedule.step(hp_max=1, epochs=micro_epochs, breaks=(0.33, 0.66), factors=(0.1, 0.1)),
    "linear"   : HPSchedule.linear(hp_max=1, epochs=micro_epochs),
    "cosine"   : HPSchedule.sgdr(hp_max=1, period_length=micro_epochs),
}

for macro_k,macro_hp in macro.items():
    for micro_k,micro_hp in micro.items():
        hp = HPSchedule.prod_schedule([
            macro_hp,
            micro_hp,
        ])
        
        hps = np.vstack([hp(i) for i in np.arange(0, macro_epochs, 0.001)])
        _ = plt.plot(hps, label='%s-%s' % (macro_k, micro_k))
    
    _ = plt.legend()
    show_plot()