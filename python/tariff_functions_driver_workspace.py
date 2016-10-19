# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:26:58 2016

@author: pgagnon

To Do:
- I have not checked the 12x24 to 8760 conversion
"""

import requests as req
import numpy as np
import pandas as pd
import tariff_functions as tFuncs

    

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
    
#%%
    
tariff_object = tFuncs.Tariff('574dbcac5457a3d3795e629f')

tariff_object.write_json('pge_e20.json')

#%%

#tariff_object.d_flat_prices = np.zeros((1,12))
#tariff_object.d_tou_prices = np.zeros((1,4))

load_profiles = np.genfromtxt('profiles_for_bill_calc.csv', delimiter=",", skip_header=1)

dispatcher_profile = load_profiles[:,0]
reopt_profile = load_profiles[:,1]
original_profile = load_profiles[:,2]

original_bill, original_results = tFuncs.bill_calculator(original_profile, tariff_object, export_tariff)

dispatcher_bill, dispatcher_results = tFuncs.bill_calculator(dispatcher_profile, tariff_object, export_tariff)
reopt_bill, reopt_results = tFuncs.bill_calculator(reopt_profile, tariff_object, export_tariff)

print "Original bill: $", original_bill
print "Dispatcher bill: $", dispatcher_bill
print "REopt bill: $", reopt_bill
print "Dispatcher bill savings: $", (original_bill-dispatcher_bill)
print "REopt bill savings: $", (original_bill-reopt_bill)
print "Percent savings error: ", 100 * (dispatcher_bill-reopt_bill) / (original_bill-reopt_bill), "%"
