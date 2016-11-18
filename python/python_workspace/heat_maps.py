# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:15:43 2016

@author: pgagnon
"""

import sys
sys.path.append('C:/users/pgagnon/desktop/support_functions/python')

import numpy as np
import tariff_functions as tFuncs
import dispatch_functions as dFuncs
import financial_functions as fFuncs
import matplotlib.pyplot as plt

analysis_years = 20
inflation = 0.02
real_d = np.array([0.08])
nom_d = (1 + real_d) * (1 + inflation) - 1

e_escalation = np.zeros(analysis_years+1)
e_escalation[0] = 1.0
e_escalation[1:] = (1.0039)**np.arange(analysis_years)

pv_price = np.array([2160.0])
inverter_price = np.array([0.0])
batt_power_price = np.array([1600.0])
batt_cap_price = np.array([500.0])
pv_om = np.array([20.0])
batt_om = np.array([0.0])

pv_deg = 0.005

batt_chg_frac = np.array([1.0])
batt_replacement_sch = np.array([10])
sector = np.array(['com'])

itc = np.array([0.3])
deprec_sched_single = np.array([0.6, .16, .096, 0.0576, 0.0576, .0288])
deprec_sched = np.zeros([1,len(deprec_sched_single)]) + deprec_sched_single

fed_tax_rate = np.array([0.35])
state_tax_rate = np.array([0.0])

debt_fraction = np.array([0.0])
loan_rate = np.array([0.05])
loan_term = np.array(20)

cash_incentives = np.array([0])
ibi = np.array([0])
cbi = np.array([0])
pbi = np.array([0])


class export_tariff:
    """
    Structure of compensation for exported generation. Currently only two 
    styles: full-retail NEM, and instantanous TOU energy value. 
    """
     
    full_retail_nem = True
    prices = np.zeros([1, 1], float)     
    levels = np.zeros([1, 1], float)
    periods_8760 = np.zeros(8760, int)
    period_tou_n = 1
        
pv_size = np.array([2088.63])
batt_cap = np.array([521.727])
batt_power = np.array([181.938])

batt = dFuncs.Battery(SOC_min=0.2, eta_charge=0.911, eta_discharge=0.911)

profile = np.genfromtxt('input_profile_lg_office_ny.csv', delimiter=",", skip_header=1)
original_load_profile = profile[:,0]
pv_cf_profile = profile[:,1]
aec = np.sum(original_load_profile)
naep = np.sum(pv_cf_profile)

#%%
if True:
    tariff_ids = ['574e067d5457a349215e629d', # Lg voluntary, any size
                  '574e04d35457a348215e629d', # Lg TOD, 1500kW min
                  '574e02195457a343795e629e'] # Lg low tension, 1500kW max
    
    tariff = tFuncs.Tariff(urdb_id='574e02195457a343795e629e')
    tariff.write_json('coned_sc9_large_low_tension.json')
#%%

tariff_file_names = ['coned_sc9_large_voluntary_tod.json',
                     'coned_sc9_large_tod.json',
                     'coned_sc9_large_low_tension.json']

original_bills = np.zeros(3)
for t, tariff_file_name in enumerate(tariff_file_names):
    tariff = tFuncs.Tariff(json_file_name=tariff_file_name)
    original_bills[t], original_bill_results = tFuncs.bill_calculator(original_load_profile, tariff, export_tariff)
    print tariff_file_name, original_bills[t]
    
# Just do the large TOD for now, for the large office, since I'm not sure it can subscribe to the others.
    
#%%
    
years = range(analysis_years)
pv_inc = 10
batt_inc = 10
pv_sizes = np.linspace(0,1000,pv_inc)
batt_powers = np.linspace(1,500,batt_inc)
batt_ratio = 2
d_inc_n = 20
DP_inc = 20
dispatched_bills = np.zeros([pv_inc, batt_inc])
fy_bill_savings = np.zeros([pv_inc, batt_inc]) 
npvs = np.zeros([pv_inc, batt_inc])
payback_periods = np.zeros([pv_inc, batt_inc])
npv_bill_savings = np.zeros([pv_inc, batt_inc])
irrs = np.zeros([pv_inc, batt_inc])
bill_savings_cf = np.zeros(analysis_years+1)
cf_results = np.zeros([pv_inc, batt_inc], dtype=object)

tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_tod.json')
original_bill, original_bill_results = tFuncs.bill_calculator(original_load_profile, tariff, export_tariff)

#%%
for p, pv_size in enumerate(pv_sizes):
    pv_profile = pv_size*pv_cf_profile
    aep = np.sum(pv_profile)
    load_profile = original_load_profile - pv_size*pv_cf_profile
    energy_penetration = aep / aec
    print "annual energy penetration:", energy_penetration
        
    for b, batt_power in enumerate(batt_powers):
        print p, b
        batt.set_cap_and_power(batt_power*batt_ratio, batt_power)
        
        for year in years:
            print year
            batt.set_cycle_deg(365.0*5/7*np.mod(year,10)) #Assume it cycles 5 times per week
    
            pv_profile = pv_size*pv_cf_profile*(1-pv_deg)**year
            load_profile = original_load_profile - pv_profile
            
            dispatch_results = dFuncs.determine_optimal_dispatch(load_profile, batt, tariff, export_tariff, d_inc_n, DP_inc, estimator_params=None, estimated=False)
            dispatched_net_profile = dispatch_results['opt_load_profile']
            
            dispatched_bill, dispatched_bill_results = tFuncs.bill_calculator(dispatched_net_profile, tariff, export_tariff)
            
            bill_savings_cf[year] = (original_bill - dispatched_bill)*e_escalation[year]
            
            fy_bill_savings[p,b] = bill_savings_cf[1]
            
            
        
        cf_result = fFuncs.cashflow_constructor(bill_savings_cf, 
                                 pv_size, pv_price, inverter_price, pv_om,
                                 batt.nameplate_cap, batt.nameplate_power, batt_power_price, batt_cap_price, batt_chg_frac,
                                 batt_replacement_sch, batt_om,
                                 sector, itc, deprec_sched, 
                                 fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
                                 analysis_years, inflation, 
                                 loan_rate, loan_term, 
                                 cash_incentives, ibi, cbi, pbi)
        cf_results[p,b] = cf_result
        
        payback_periods[p,b] = fFuncs.calc_payback_vectorized(cf_result['cf'], analysis_years)
        npvs[p,b] = cf_result['npv']
#        npv_bill_savings[p,b] = fFuncs.calc_npv(np.array(cf_result['bill_savings']), nom_d)  
        irrs[p,b] = fFuncs.virr(cf_result['cf'])                   
                   
#%%                   
#np.save('cf_results.npy', cf_results, allow_pickle=True)
#np.save('npvs.npy', npvs, allow_pickle=True)
#np.save('fy_bill_savings.npy', fy_bill_savings, allow_pickle=True)
#np.save('irrs.npy', irrs, allow_pickle=True)



#%%
                                 
                                 
# PV is Y, Battery is X
X, Y = np.meshgrid(batt_powers*batt_ratio, pv_sizes)

#NPV
fig1 = plt.figure(1, figsize=(5,5))
plt.contourf(X, Y, npvs) #frac_self_supply_2d
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Sizes (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Capacities (kWh)', size=14)
plt.title('NPV')

#Savings                     
fig2 = plt.figure(2, figsize=(5,5))
plt.contourf(X, Y, fy_bill_savings) #frac_self_supply_2d
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Sizes (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Capacities (kWh)', size=14)
plt.title('First Year Bill Savings')

#IRR
irrs_clipped = np.clip(irrs,0,1)
fig3 = plt.figure(3, figsize=(5,5))
plt.contourf(X, Y, irrs_clipped) #frac_self_supply_2d
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Sizes (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Capacities (kWh)', size=14)
plt.title('Project IRRs')

#%%
# plot of the reduction beyond the simple addition of the individual batt and PV potential
reduction_above_sum_potential = np.zeros((pv_inc,batt_inc))
for p, pv_size in enumerate(pv_sizes):
    for b, batt_power in enumerate(batt_powers):
        reduction_above_sum_potential[p,b] = fy_bill_savings[p,b] / (fy_bill_savings[0,b]+fy_bill_savings[p,0])
reduction_above_sum_potential[0,0] = 1.0
plt.figure(4, figsize=(5,5))
plt.contourf(X, Y, reduction_above_sum_potential)
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Size (kW)', size=14)
plt.title('S+S Cooperation Ratio = \n\n(demand reduction of combined solar+storage systems)\n---------------------------------------------------------------------------\n(reduction of solar)+(reduction of storage)')
#plt.title('S+S Cooperation Ratio:\n' r'$\frac{\mathrm{demand reduction of combined Solar+Storage systems}}{(reduction of solar)+(reduction of storage)}$',
#         fontsize=14)

#%%
# Plot of NPV with IRR contours overlaid

# PV is Y, Battery is X
X, Y = np.meshgrid(batt_powers*batt_ratio, pv_sizes)

#NPV
plt.figure(5, figsize=(5,5))
fig1 = plt.contourf(X, Y, npvs)
plt.colorbar()

levels = np.array([0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25])
fig1 = plt.contour(X, Y, irrs, 
                   levels=levels,
                   colors='black')
plt.clabel(fig1, inline=1, fontsize=10)
plt.grid(True)
plt.ylabel('PV Sizes (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Capacities (kWh)', size=14)
plt.title('NPV with IRR overlay')

