# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:55:13 2021

@author: siegfriedkaidisch

equidistant grid
"""

import numpy as np
import pandas as pd

#grid = {'qsquared': np.round(np.linspace(0,10,64), decimals=1)}
grid = {'qsquared': np.linspace(0,1e5,64)}
pd.DataFrame(grid).to_csv("integration_gridpoints.csv", index=None)