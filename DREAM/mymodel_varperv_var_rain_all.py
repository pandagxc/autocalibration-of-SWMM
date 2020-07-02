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
import subprocess as sup
#import matplotlib.pyplot as plt

events = ['20180530', '20180622', '20180625', '20180713', '20180724', '20180828', '20180831', '20190825']
#events = ['20180530', '20180622', '20180625', '20180713', '20180724', '20180828']
path_obs = '/mnt/c/Users/999ga/OneDrive/博士课题/毕业论文/swmm_model/dream_swmm/runofffile'
path_inp = '/mnt/c/Users/999ga/OneDrive/博士课题/毕业论文/swmm_model/dream_swmm/inpfile'
os.chdir(path_obs)
obs_list = []
for event in events:
    runoff_file = 'runoff' + event + '.txt'
    obs = pd.read_csv(runoff_file, header=None)
    obs = obs[0].values
    obs_list = obs_list + list(obs)
#obs = pd.read_csv('runoff20180530.txt', header=None)
#obs = pd.read_csv('runoff_obs.txt', header=None)
#obs = obs[0].values
os.chdir(path_inp)


def model_gen():
    varlist = []
    # all the paramters are multiplied by 100 to adapt to the distribution fuction.
    # remeber to divide by 100 when used in the model.
    rain = []
    k = pymc.Uniform('k', lower=0.2, upper=5)                    # multiplier of characteristic width. [0.2, 5] 
    pctImperv_c = pymc.Uniform('pctImperv_c', lower=0.50, upper=0.80)
    pctImperv_l = pymc.Uniform('pctImperv_l', lower=0.05, upper=0.20)
    slope = pymc.Uniform('slope', lower=0.0, upper=0.03)               # slope of subcatchment. [0, 0.01]
    nPerv = pymc.Uniform('nPerv', lower=0.03, upper=0.80)
    nImperv = pymc.Uniform('nImperv', lower=0.01, upper=0.05)
    stoPerv = pymc.Uniform('stoPerv', lower=2, upper=8)
    stoImperv = pymc.Uniform('stoImperv', lower=1, upper=3)
    infi = pymc.Uniform('infi', lower=20, upper=200)              # Sat. hyd. conductivity. [0.5, 5]
    imd = pymc.Uniform('imd', lower=0.20, upper=0.50)                 # max sat. deficit range from [0.2, 0.5]
    #rain = pymc.Uniform('rain', lower=50, upper=150)                 # max sat. deficit range from [0.2, 0.5]
    #k = pymc.Uniform('k', lower=0, upper=500)                    # multiplier of characteristic width. [0.2, 5] 
                                                                  # width = k * sqrt(area)
    #slope = []
    #for i in range(32):
    #    slope.append(pymc.Uniform('slope'+str(i), lower=1, upper=4))

    #pctImperv_c = pymc.Uniform('pctImperv_c', lower=50, upper=80)
    #pctImperv_l = pymc.Uniform('pctImperv_l', lower=5, upper=20)
    #slope = pymc.Uniform('slope', lower=1, upper=3)               # slope of subcatchment. [0, 0.01]
    #nPerv = pymc.Uniform('nPerv', lower=3, upper=80)
    #nImperv = pymc.Uniform('nImperv', lower=1, upper=3)
    #stoPerv = pymc.Uniform('stoPerv', lower=200, upper=800)
    #stoImperv = pymc.Uniform('stoImperv', lower=100, upper=300)
    #infi = pymc.Uniform('infi', lower=100, upper=20000)              # Sat. hyd. conductivity. [0.5, 5]
    #imd = pymc.Uniform('imd', lower=20, upper=50)                 # max sat. deficit range from [0.2, 0.5]
    #rain_sig = pymc.Uniform('r_sig', lower=0.5, upper=8)          # variance for rain multiplier
    for i in range(len(events)):
        #rain.append(pymc.TruncatedNormal('rain'+str(i), mu=1, tau=rain_sig, a=0.5, b=2.0))
        rain.append(pymc.Uniform('rain'+str(i), lower=0.50, upper=1.5))
    #rain = pymc.Uniform('rain', lower=50, upper=150)                 # max sat. deficit range from [0.2, 0.5]
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
    #varlist.append(rain)
    for i in range(len(rain)):
        varlist.append(rain[i])
    #varlist.append(rain_sig)
    varlist.append(sig)

    #for i in range(len(slope)):
    #    varlist.append(slope[i])

    # run swmm model and return simulated runoff hydrograph
    @pymc.deterministic
    #def swmm_run(k=k, nPerv=nPerv, pctImperv_c=0.75, nImperv=nImperv, stoPerv=stoPerv, stoImperv=stoImperv, infi=infi, imd=imd, rain=rain):
    #def swmm_run(k=k, nPerv=nPerv, pctImperv_c=pctImperv_c, nImperv=nImperv, stoPerv=stoPerv, stoImperv=stoImperv, infi=infi, imd=imd, rain=rain):
    def swmm_run(k=k, pctImperv_c=pctImperv_c, pctImperv_l=pctImperv_l, 
                 slope=slope, nPerv=nPerv, nImperv=nImperv, stoPerv=stoPerv, 
                 stoImperv=stoImperv, infi=infi,imd=imd, rain=rain):
    #def swmm_run(k=k, slope=slope):
        #sublist = [] 
        lawn = ['s31', 's32', 's33', 's34', 's41', 's42', 's43', 's59', 's60', 's61', 's62', 's63', 's64']
        simflow = []
        for j in range(len(events)):
            inpfile = 'tianheful_backup' + events[j] + '.inp'

            i = 0
            flow_temp = 0
            with Simulation(inpfile) as sim:
            #with Simulation('tianheful_backup1.inp') as sim:
                subcatchments = Subcatchments(sim)
                #for sub in subcatchments:
                #    sublist.append(sub.subcatchmentid)
                for sub in subcatchments:
                    sub.width = k *  np.sqrt(sub.area * 10000)
                    sub.percent_impervious = pctImperv_c
                    sub.slope = slope
                    sub.perv_n = nPerv
                    sub.imperv_n = nImperv
                    sub.perv_sto = stoPerv
                    sub.imperv_sto = stoImperv
                    sub.Ks = infi
                    sub.IMD = imd
                    #sub.rainMulti = rain[j] / 100
                    sub.rainMulti = rain[j]
                #for j in range(len(sublist)):
                #    subcatchments[sublist[j]].width = k / 100 * np.sqrt(subcatchments[sublist[j]].area * 10000)
                #    subcatchments[sublist[j]].slope = slope[j] / 100
                #    subcatchments[sublist[j]].Ks = infi / 100
                #    subcatchments[sublist[j]].IMD = imd / 100
                for sublawn in lawn:
                    subcatchments[sublawn].percent_impervious = pctImperv_l
                    #subcatchments[sublawn].percent_impervious = 0.05
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

        sup.run('rm *.out *.rpt', shell=True)
        return simflow
        #return [k, slope, infi, imd]
    
    #results = pymc.Normal('results', mu=swmm_run, tau=1, value=obs, observed=True)
    results = pymc.Normal('results', mu=swmm_run, tau=sig, value=obs_list, observed=True)
    varlist.append(results)
    #print(len(varlist))
    return varlist

#model_gen()
#print('\n\t')
#print(model_gen()[-1].logp)
#print(model_gen()[4].logp)
#print(model_gen()[0].logp)
#print(model_gen()[1].logp)
#print(model_gen()[1].logp)
#print(model_gen()[4].observed)
