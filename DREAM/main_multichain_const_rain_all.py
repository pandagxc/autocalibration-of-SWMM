#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.append('/home/panda/.local/lib/python3.6/site-packages/multichain_mcmc')
sys.path.append('/mnt/c/Users/999ga/OneDrive/博士课题/毕业论文/swmm_model/dream_swmm')
import pymc
import multichain_mcmc
#import mymodel_variance
#import mymodel_varslope
#import mymodel_var_rain
#import mymodel_varant_sig
import mymodel_varperv_const_rain_all
#import mymodel_varperv_ar
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

path_parms = '/mnt/c/Users/999ga/OneDrive/博士课题/毕业论文/swmm_model/dream_swmm/parameters'
#sampler = multichain_mcmc.DreamSampler(mymodel_varslope.model_gen)
#sampler.sample(nChains=40, ndraw=5000, ndraw_max=300000)
#sampler = multichain_mcmc.DreamSampler(mymodel_var_rain.model_gen)
#sampler = multichain_mcmc.DreamSampler(mymodel_variance.model_gen)  # 5 params
sampler = multichain_mcmc.DreamSampler(mymodel_varperv_const_rain_all.model_gen)  # 11 params
#sampler.sample(nChains=10, ndraw=3000)
sampler.sample(nChains=22, ndraw=10000, ndraw_max=1000000)
print(sampler.R)
history = sampler.history
#history = history[-3000:, :]
os.chdir(path_parms)
np.savetxt('params_const_rain_all_7events_10p.txt', history, fmt='%f', delimiter=',')
#np.savetxt('params828_11p_r2.txt', history, fmt='%f', delimiter=',')
#np.savetxt('params_varslope_5k.txt', history, fmt='%f', delimiter=',')
#print(type(history))
print(history.shape)
print(sampler.accepts_ratio)
print(sampler.time)
#fig = plt.figure()
#ax1 = plt.subplot(2, 2, 1)
#plt.hist(history[:, 0])
#ax2 = plt.subplot(2, 2, 2)
#plt.hist(history[:, 1])
#ax3 = plt.subplot(2, 2, 3)
#plt.hist(history[:, 2])
#ax3 = plt.subplot(2, 2, 4)
#plt.hist(history[:, 3])
#plt.show()
