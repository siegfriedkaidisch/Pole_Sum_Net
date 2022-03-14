#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg, siegfriedkaidisch

Prepare data for application

"""
import pandas as pd
from scipy import interpolate
import numpy as np
import os

from parameters import data_dir

def fun_sigma_S(A, B, p2):
    # A, B and p2 must be ndarrays of shape (n,)
    return A/( A**2*p2 + B**2 )

def fun_sigma_V(A, B, p2):
    # A, B and p2 must be ndarrays of shape (n,)
    return B/( A**2*p2 + B**2 )

### Import data
data = pd.read_pickle(os.path.join(data_dir, 'data.pkl'))
p2 = data["data"]["p2"]
A = data["data"]["A"]
B = data["data"]["B"]

### Calculate sigma_S/V
sigma_S = fun_sigma_S(A, B, p2)
sigma_V = fun_sigma_V(A, B, p2)
#plt.plot(p2, sigma_S
#plt.plot(p2, sigma_V)
#plt.xscale('log')
#plt.show()

### Interpolate to our p2 grid (take log of x-axis -> interpolate -> take exp)
my_p2 = pd.read_csv(os.path.join(data_dir, "integration_gridpoints.csv")).to_numpy().reshape(-1)
fun_interpolate_sigma_S = interpolate.interp1d(np.log(p2), sigma_S, kind='linear')
my_sigma_S = fun_interpolate_sigma_S(np.log(my_p2))
fun_interpolate_sigma_V = interpolate.interp1d(np.log(p2), sigma_V, kind='linear')
my_sigma_V = fun_interpolate_sigma_V(np.log(my_p2))

#plt.plot(p2, sigma_S)
#plt.plot(my_p2, my_sigma_S)
#plt.plot(p2, sigma_V)
#plt.plot(my_p2, my_sigma_V)
#plt.xscale('log')
#plt.show()