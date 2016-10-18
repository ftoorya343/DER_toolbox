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

    

#%%
class export_tariff:
    """
    Structure of compensation for exported generation. Currently only two 
    styles: full-retail NEM, and instantanous TOU energy value. 
    """
     
    full_retail_nem = True
    prices = np.zeros([1, 1], int)     
    levels = np.zeros([1, 1], int)
    periods_8760 = np.zeros(8760, int)
    period_tou_n = 1
    
#%%
    
tariff_object = tFuncs.Tariff('57bcd2b65457a3a67e540154')

tariff_object.write_json('dummy_tariff.json')

#tariff2 = tFuncs.Tariff(json_file_name='dummy_tariff.json')

#load_profile = np.genfromtxt('large_office_profile.csv')
#pv_profile = np.genfromtxt('pv_profile.csv')
#
#net_profile = load_profile - 500*pv_profile
#
#bill, results = tFuncs.bill_calculator(net_profile, tariff_object, export_tariff)
#
#
#print "Total bill: $", bill
