# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 16:13:43 2016

@author: pgagnon
"""

import numpy as np
import pandas as pd
import demand_functions as demFun
import rate_import_functions as rate_import
import matplotlib.pyplot as plt
import * from tariff_generator_and_bill_calc_workspace
     
    
#%%
def irr_calc(values):
    """
    Return the minimum Internal Rate of Return (IRR) within the range [-30%,+Inf].

    This is the "average" periodically compounded rate of return
    that gives a net present value of 0.0; for a more complete explanation,
    see Notes below.

    Parameters
    ----------
    values : array_like, shape(N,)
        Input cash flows per time period.  By convention, net "deposits"
        are negative and net "withdrawals" are positive.  Thus, for example,
        at least the first element of `values`, which represents the initial
        investment, will typically be negative.

    Returns
    -------
    out : float
        Internal Rate of Return for periodic input values.

    Notes
    -----

    """
    res = np.roots(values[::-1])
    # Find the root(s) between 0 and 1
    mask = (res.imag == 0) & (res.real > 0)
    res = res[mask].real
    if res.size == 0:
        return np.nan
    rate = 1.0/res - 1
    if sum(values)>0:
        rate = rate[rate>0] # Negative IRR is returned otherwise
    rate = rate[rate>-.3]
    if rate.size == 0:
        return np.nan
    if rate.size == 1:
        rate = rate.item()
    else: 
        rate = min(rate)
    return rate
    
#%%
def calc_npv(cfs,dr):
    ''' Vectorized NPV calculation based on (m x n) cashflows and (n x 1) 
    discount rate
    
    IN: cfs - numpy array - project cash flows ($/yr)
        dr  - numpy array - annual discount rate (decimal)
        
    OUT: npv - numpy array - net present value of cash flows ($) 
    
    '''
    dr = dr[:,np.newaxis]
    tmp = np.empty(cfs.shape)
    tmp[:,0] = 1
    tmp[:,1:] = 1/(1+dr)
    drm = np.cumprod(tmp, axis = 1)        
    npv = (drm * cfs).sum(axis = 1)   
    return npv
    

    
#%% Analysis of a 2D design space for a single building
    
######################## Define Directories ###################################
load_profile_dir = "C:\Users\pgagnon\Desktop\dGen\S+S Project\Python_workspace\NYSERDA_workspace"
PV_profile_dir = "C:\Users\pgagnon\Desktop\dGen\S+S Project\Python_workspace\NYSERDA_workspace"

######################## Import Tariff ###################################
tariff = Tariff('574e03045457a30b7e5e62a0')

#################### Set details of this analysis ########################
inc = 5
pv_sizes = np.linspace(0, 1700, inc)
batt_powers = np.linspace(0, 1000, inc)
batt_ratio = 3
d_inc_n = 50 # Number of demand levels between original and lowest possible that will be explored
demand_charge = 20 # $/kW
pv_cost = 3000
batt_kW_cost = 800
batt_kWh_cost = 250
ITC = 0.3
MACRS = np.array([.2, .32, .192, .1152, .1152, .0576])
tax_rate = 0.3
discount = 0.1

class export_tariff:
    """
    Structure of compensation for exported generation. Currently only two 
    styles: full-retail NEM, and instantanous TOU energy value. 
    """
     
    full_retail_nem = False
    prices = np.zeros([1, 1], int)     
    levels = np.zeros([1, 1], int)
    periods_8760 = np.zeros(8760, int)
    period_n = 1

class batt:
    eta = 0.95 # battery half-trip efficiency
    cap = 0
    power = 0

load_temp = pd.read_csv('large_office_profile.csv')
pv_temp = pd.read_csv('pv_profile.csv')

load_profile = np.array(load_temp)
pv_cf_profiles = np.array(pv_temp)

# Calculate mean peak demand for bld with no system
results_base = bill_calculator(load_profile, tariff, export_tariff)
mean_demand_base = results_base['period_maxs_unratcheted'].mean()

mean_demand_sys = np.zeros((inc,inc))
mean_demand_reduction = np.zeros((inc,inc))
annual_bill_savings = np.zeros((inc,inc))
system_cost = np.zeros((inc,inc))
irr = np.zeros((inc,inc))
ITC_value = np.zeros((inc,inc))
npvs = np.zeros((inc,inc))


for p, pv_size in enumerate(pv_sizes):
    for b, batt_power in enumerate(batt_powers):
        print p, b
        results_storage = np.zeros((n_years,12))
        results_pv_and_storage = np.zeros((n_years,12))
        batt.cap = batt_power * 3
        batt.power = batt_power
        
        pv_profile = pv_cf_profiles * pv_size
        load_and_pv_profile = load_profile - pv_profile
        
        dispatched_results = bill_calculator(load_and_pv_profile, tariff, export_tariff)
        results_pv_and_storage[year,:] = results_pv_and_storage_year['cheapest_possible_demands'].reshape(tariff.d_n*12)

        mean_demand_sys[p,b] = results_pv_and_storage.mean()
        mean_demand_reduction[p,b] = mean_demand_base - results_pv_and_storage.mean()
        annual_bill_savings[p,b] = mean_demand_reduction[p,b]*demand_charge*12
        system_cost[p,b] = (pv_size*pv_cost + batt.cap*batt_kWh_cost + batt.power*batt_kW_cost)
        ITC_value[p,b] = system_cost[p,b]*ITC*tax_rate
        cf = np.zeros(25)
        cf[:] = annual_bill_savings[p,b]
        cf[0] = -system_cost[p,b]
        cf[1] = ITC_value[p,b]
        cf[1:len(MACRS)] = system_cost[p,b]*MACRS*tax_rate
        irr[p,b] = irr_calc(cf)
        npvs[p,b] = calc_npv(cf,discount)


#%%
X, Y = np.meshgrid(batt_powers, pv_sizes) # PV is Y, Battery is X
plt.figure(1, figsize=(5,5))
plt.contourf(X, Y, mean_demand_reduction)
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Size (kW)', size=14)
plt.title('Mean Reduction in Monthly Peak Demand\nLarge Office Building, Peak Demand = %d kW\n1 MW PV = 0.22 Annual Energy Penetration' % np.max(load_profile))

#%%
# plot of the reduction beyond the simple addition of the individual batt and PV potential
reduction_above_sum_potential = np.zeros((inc,inc))
for p, pv_size in enumerate(pv_sizes):
    for b, batt_power in enumerate(batt_powers):
        reduction_above_sum_potential[p,b] = mean_demand_reduction[p,b] / (mean_demand_reduction[0,b]+mean_demand_reduction[p,0])
reduction_above_sum_potential[0,0] = 1.0
plt.figure(2, figsize=(5,5))
plt.contourf(X, Y, reduction_above_sum_potential)
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Size (kW)', size=14)
plt.title('S+S Cooperation Ratio = \n\n(demand reduction of combined solar+storage systems)\n---------------------------------------------------------------------------\n(reduction of solar)+(reduction of storage)')
#plt.title('S+S Cooperation Ratio:\n' r'$\frac{\mathrm{demand reduction of combined Solar+Storage systems}}{(reduction of solar)+(reduction of storage)}$',
#         fontsize=14)
#%%
# plot of IRR
plt.figure(3, figsize=(5,5))
plt.contourf(X, Y, irr)
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Size (kW)', size=14)
plt.title('IRRs')