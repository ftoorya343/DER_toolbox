# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 23:44:32 2016

@author: pgagnon

Battery Dispatch Driver and Plotter
"""

import numpy as np
import tariff_functions as tFuncs
import dispatch_functions as dFuncs
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt, mpld3

full_retail_nem = True
export_tariff = tFuncs.Export_Tariff(full_retail_nem)
tariff = tFuncs.Tariff(json_file_name='sc9_rate1.json') # Rate I (small)
tar = tariff.__dict__
#tariff.coincident_peak_exists = False

batt = dFuncs.Battery(nameplate_cap=3000.0, nameplate_power=1000.0)


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

results_con = dFuncs.determine_optimal_dispatch(load_profile, pv_profile, batt, tariff, export_tariff, restrict_charge_to_pv_gen=True)
bill_con, _ = tFuncs.bill_calculator(results_con['load_profile_under_dispatch'], tariff, export_tariff)
print "constrained bill without cap", results_con['bill_under_dispatch']
print "constrained bill with cap", bill_con


results_uncon = dFuncs.determine_optimal_dispatch(load_profile, pv_profile, batt, tariff, export_tariff, restrict_charge_to_pv_gen=False)
bill_uncon, _ = tFuncs.bill_calculator(results_uncon['load_profile_under_dispatch'], tariff, export_tariff)
print "unconstrained bill without cap",results_uncon['bill_under_dispatch']
print "unconstrained bill with cap", bill_uncon

#hours = np.arange(4050, 4180)
hours = range(8760)
plt.figure(1, figsize=(7,5))
plt.plot(hours, load_profile[hours])
plt.plot(hours, load_and_pv_profile[hours])
plt.plot(hours, pv_profile[hours])
plt.plot(hours, results_uncon['load_profile_under_dispatch'][hours])
plt.axis([4060, 4080, 1000, 1800])
plt.legend(['Original load profile', 'load profile with PV only', 'pv profile',  'load profile with PV and battery'])
plt.grid(True)
plt.xlabel('Hour of the year')
plt.ylabel('Electric\nDemand\n(kW)', rotation=0, labelpad=30)
#mpld3.show()

plt.figure(2, figsize=(8,5))
plt.plot(hours, load_profile[hours])
plt.plot(hours, load_and_pv_profile[hours])
plt.plot(hours, pv_profile[hours])
plt.plot(hours, results_uncon['load_profile_under_dispatch'][hours])
plt.axis([4050, 4180, 0, 2200])
plt.legend(['Original load profile', 'load profile with PV only', 'pv profile',  'load profile with PV and battery'])
plt.grid(True)
plt.xlabel('Hour of the year')
plt.ylabel('Electric\nDemand\n(kW)', rotation=0, labelpad=30)
