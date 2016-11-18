# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:03:29 2016

@author: pgagnon
"""

import sys
sys.path.append('C:/users/pgagnon/desktop/support_functions/python')
import numpy as np
import os
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt, mpld3
import pandas as pd
import tariff_functions as tFuncs

profiles = pd.read_csv('profiles_to_compare.csv')

#%%
      
#tariff_object = tFuncs.Tariff('574e067d5457a349215e629d')
#tariff_object.write_json('coned_sc9_large_voluntary_tod.json')
tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_voluntary_tod.json')


class export_tariff:
    """
    Structure of compensation for exported generation. Currently only two 
    styles: full-retail NEM, and instantanous TOU energy value. 
    """
     
    full_retail_nem = False
    prices = np.zeros([1, 1], float)     
    levels = np.zeros([1, 1], float)
    periods_8760 = np.zeros(8760, int)
    period_tou_n = 1


original_bill, original_results = tFuncs.bill_calculator(np.array(profiles['original_load']), tariff, export_tariff)
dispatcher_bill, dispatcher_results = tFuncs.bill_calculator(np.array(profiles['python_dispatcher']), tariff, export_tariff)
reopt_bill, reopt_results = tFuncs.bill_calculator(np.array(profiles['reopt_grid_to_load']), tariff, export_tariff)

print "Original bill: $", original_bill
print "Dispatcher bill: $", dispatcher_bill
print "REopt bill: $", reopt_bill
print "Dispatcher bill savings: $", (original_bill-dispatcher_bill)
print "REopt bill savings: $", (original_bill-reopt_bill)
print "Dispatcher vs REopt Savings: ", 100 * (original_bill-dispatcher_bill) / (original_bill-reopt_bill), "%"
print "Dispatcher savings error: ", 100 * (dispatcher_bill-reopt_bill) / (original_bill-reopt_bill), "%"


#%% Plot
#hours = range(8760)
#plt.figure(1, figsize=(17,5))
#plt.plot(hours, profiles['net_load'])
#plt.plot(hours, profiles['reopt_grid_to_load'])
#plt.plot(hours, profiles['python_dispatcher'])
#plt.bar(hours, tariff.d_tou_8760*np.max(profiles['original_load'])/np.max(tariff.d_tou_8760), alpha=0.2, width=1)
#plt.legend(['load profile with PV only', 'REopt dispatch', 'python dispatch'])
#plt.grid(True)
#plt.xlabel('Hour of the year')
#plt.ylabel('Electric\nDemand\n(kW)', rotation=0, labelpad=30)
#mpld3.show()