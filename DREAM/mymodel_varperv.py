#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/panda/.local/lib/python3.6/site-packages/multichain_mcmc')
import pymc
import os
import numpy as np
import pandas as pd
from scipy import stats
from pyswmm import Simulation, Subcatchments, Links
#import matplotlib.pyplot as plt

path_obs = '/mnt/c/Users/999ga/OneDrive/博士课题/毕业论文/swmm_model/dream_swmm/runofffile'
path_inp = '/mnt/c/Users/999ga/OneDrive/博士课题/毕业论文/swmm_model/dream_swmm/inpfile'
os.chdir(path_obs)
obs = pd.read_csv('runoff625_obs.txt', header=None)
#obs = pd.read_csv('runoff_obs.txt', header=None)
obs = obs[0].values
os.chdir(path_inp)


def model_gen():
    varlist = []
    # all the paramters are multiplied by 100 to adapt to the distribution fuction.
    # remeber to divide by 100 when used in the model.
    k = pymc.Uniform('k', lower=20, upper=500)                    # multiplier of characteristic width. [0.2, 5] 
                                                                  # width = k * sqrt(area)
    #slope = []
    #for i in range(32):
    #    slope.append(pymc.Uniform('slope'+str(i), lower=1, upper=4))

    pctImperv_c = pymc.Uniform('pctImperv_c', lower=50, upper=70)
    pctImperv_l = pymc.Uniform('pctImperv_l', lower=5, upper=20)
    slope = pymc.Uniform('slope', lower=1, upper=3)               # slope of subcatchment. [0, 0.01]
    nPerv = pymc.Uniform('nPerv', lower=1, upper=3)
    nImperv = pymc.Uniform('nImperv', lower=4, upper=12)
    stoPerv = pymc.Uniform('stoPerv', lower=0, upper=50)
    stoImperv = pymc.Uniform('stoImperv', lower=0, upper=30)
    infi = pymc.Uniform('infi', lower=10, upper=150)              # Sat. hyd. conductivity. [0.5, 5]
    imd = pymc.Uniform('imd', lower=20, upper=50)                 # max sat. deficit range from [0.2, 0.5]
    rain = pymc.Uniform('rain', lower=50, upper=150)                 # max sat. deficit range from [0.2, 0.5]
    sig = pymc.Uniform('sig', lower=1, upper=100)                 # variance of the distribution of object function
    
    varlist.append(k)
    varlist.append(pctImperv_c)
    varlist.append(pctImperv_l)
    varlist.append(slope)
    varlist.append(nPerv)
    varlist.append(nImperv)
    varlist.append(stoPerv)
    varlist.append(stoImperv)
    varlist.append(infi)
    varlist.append(imd)
    varlist.append(rain)
    varlist.append(sig)

    #for i in range(len(slope)):
    #    varlist.append(slope[i])

    # run swmm model and return simulated runoff hydrograph
    @pymc.deterministic
    def swmm_run(k=k, pctImperv_c=pctImperv_c, pctImperv_l=pctImperv_l, 
                 slope=slope, nPerv=nPerv, nImperv=nImperv, stoPerv=stoPerv, 
                 rain=rain, stoImperv=stoImperv, infi=infi,imd=imd):
    #def swmm_run(k=k, slope=slope):
        #sublist = [] 
        lawn = ['s31', 's32', 's33', 's34', 's41', 's42', 's43', 's59', 's60', 's61', 's62', 's63', 's64']
        simflow = []
        i = 0
        flow_temp = 0
        with Simulation('tianheful_backup625.inp') as sim:
        #with Simulation('tianheful_backup1.inp') as sim:
            subcatchments = Subcatchments(sim)
            #for sub in subcatchments:
            #    sublist.append(sub.subcatchmentid)
            for sub in subcatchments:
                sub.width = k / 100  *  np.sqrt(sub.area * 10000)
                sub.percent_impervious = pctImperv_c / 100
                sub.slope = slope / 100
                sub.perv_n = nPerv / 100
                sub.imperv_n = nImperv / 100
                sub.perv_sto = stoPerv / 100
                sub.imperv_sto = stoImperv / 100
                sub.Ks = infi / 100
                sub.IMD = imd / 100
                sub.rainMulti = rain / 100
            #for j in range(len(sublist)):
            #    subcatchments[sublist[j]].width = k / 100 * np.sqrt(subcatchments[sublist[j]].area * 10000)
            #    subcatchments[sublist[j]].slope = slope[j] / 100
            #    subcatchments[sublist[j]].Ks = infi / 100
            #    subcatchments[sublist[j]].IMD = imd / 100
            for sublawn in lawn:
                subcatchments[sublawn].percent_impervious = pctImperv_l / 100
                #print(sublawn)
                #print(subcatchments[sublawn].percent_impervious)
            
            links = Links(sim)
            #for link in links:
            #   link.con_roughness = roughness / 100
            l29 = links['l29']
            #print(l29.con_roughness)
            for step in sim:
                flow_temp += l29.flow
                #  integrate 30s data to 1min
                i += 1
                if i > 1:
                    #print(i)
                    simflow.append(flow_temp / 2)
                    i = 0
                    flow_temp = 0
                #simflow.append(l29.flow)
                #print(sim.current_time)
            #print(sim.flow_routing_error)

        return simflow
        #return [k, slope, infi, imd]
    
    #results = pymc.Normal('results', mu=swmm_run, tau=1, value=obs, observed=True)
    results = pymc.Normal('results', mu=swmm_run, tau=sig, value=obs, observed=True)
    varlist.append(results)
    #print(len(varlist))
    return varlist

#model_gen()
#print('\n\t')
#print(model_gen()[1].logp)
#print(model_gen()[4].logp)
#print(model_gen()[0].logp)
#print(model_gen()[1].logp)
#print(model_gen()[1].logp)
#print(model_gen()[4].observed)
