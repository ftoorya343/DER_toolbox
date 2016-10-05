# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 23:44:32 2016

@author: pgagnon

Battery Dispatch Driver and Plotter
"""

import numpy as np
import os
import storage_functions_pieter as storFuncP
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt, mpld3


tariff = storFuncP.import_tariff('MGS_tariff_delmarva.csv')
class NEM_tariff:
    style = 0 # style: 0 = full retail, 1 = TOU schedule
    e_credits = tariff.e_prices_no_tier #Intended to replicate the normal tariff e charges per period, but I could change this to an 8760 as well
class batt:
    eta = 0.90 # battery half-trip efficiency
    power = 100.0
    cap = power*4

profile = np.genfromtxt('input_profile_lg_office_delaware.csv', delimiter=",", skip_header=1)
load_profile = profile[:,0]
pv_cf_profile = profile[:,1]

pv_size = 250.0
load_and_pv_profile = load_profile - pv_size*pv_cf_profile
pv_profile = pv_size*pv_cf_profile
aep = np.sum(pv_profile)
aec = np.sum(load_profile)
energy_penetration = aep / aec
print "annual energy penetration:", energy_penetration

estimator_params = storFuncP.calc_estimator_params(load_and_pv_profile, tariff, NEM_tariff, batt.eta)
estimator_toggle = False
d_inc_n = 100

results_dict = storFuncP.calc_bill_under_optimal_dispatch(tariff, NEM_tariff, load_and_pv_profile, batt, estimator_toggle, estimator_params, d_inc_n)

#hours = np.arange(4050, 4180)
hours = range(8760)
plt.figure(1, figsize=(7,5))
plt.plot(hours, load_profile[hours])
plt.plot(hours, load_and_pv_profile[hours])
plt.plot(hours, results_dict['net_profile'][hours])
plt.axis([4060, 4080, 1000, 1800])
plt.legend(['Original load profile', 'load profile with PV only', 'load profile with PV and battery'])
plt.grid(True)
plt.xlabel('Hour of the year')
plt.ylabel('Electric\nDemand\n(kW)', rotation=0, labelpad=30)
#mpld3.show()

plt.figure(2, figsize=(8,5))
plt.plot(hours, load_profile[hours])
plt.plot(hours, load_and_pv_profile[hours])
plt.plot(hours, results_dict['net_profile'][hours])
plt.axis([4050, 4180, 0, 2200])
plt.legend(['Original load profile', 'load profile with PV only', 'load profile with PV and battery'])
plt.grid(True)
plt.xlabel('Hour of the year')
plt.ylabel('Electric\nDemand\n(kW)', rotation=0, labelpad=30)
