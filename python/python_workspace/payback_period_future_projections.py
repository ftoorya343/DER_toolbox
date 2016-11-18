# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:38:28 2016

@author: pgagnon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:37:22 2016

@author: pgagnon
"""

import sys
sys.path.append('C:/users/pgagnon/desktop/support_functions/python')

import numpy as np
import pandas as pd
import tariff_functions as tFuncs
import dispatch_functions as dFuncs
import financial_functions as fFuncs
import matplotlib.pyplot as plt

analysis_years = 20
inflation = 0.02
real_d = np.array([0.08])
nom_d = (1 + real_d) * (1 + inflation) - 1
inflation_adjustment = (1+inflation)**np.arange(analysis_years+1)

hottest_day_index = 4069 #June 19th, 11:30AM

e_escalation = np.zeros(analysis_years+1)
e_escalation[0] = 1.0
e_escalation[1:] = (1.0039)**np.arange(analysis_years)

pv_size = np.array([500.0])
batt_cap = np.array([500.0])
batt_power = np.array([250.0])

inverter_price = np.array([0.0])
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
    
#%%

price_project_df = pd.read_csv('price_projections.csv', index_col='year')

#%%    
#tariff_object = tFuncs.Tariff('574e067d5457a349215e629d')
#tariff_object.write_json('coned_sc9_large_voluntary_tod.json')
    
tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_tod.json')

batt = dFuncs.Battery(nameplate_cap=batt_cap[0], nameplate_power=batt_power[0])

profile = np.genfromtxt('input_profile_lg_office_ny.csv', delimiter=",", skip_header=1)
original_load_profile = profile[:,0]
pv_cf_profile = profile[:,1]
aec = np.sum(original_load_profile)
naep = np.sum(pv_cf_profile)

pv_profile = pv_size*pv_cf_profile
load_profile = original_load_profile - pv_profile
aep = np.sum(pv_profile)
energy_penetration = aep / aec
print "annual energy penetration:", energy_penetration

d_inc_n = 50
DP_inc = 50

original_bill, original_bill_results = tFuncs.bill_calculator(original_load_profile, tariff, export_tariff)

years = range(analysis_years)
ref_bill_savings_cf = np.zeros(analysis_years+1)

for year in years:
    batt.set_cycle_deg(365*5/7*np.mod(year,10)) #Assume it cycles 5 times per week
    
    pv_profile = pv_size*pv_cf_profile*(1-pv_deg)**year
    load_profile = original_load_profile - pv_profile
    
    print year, "cap deg:", np.round(batt.effective_cap/batt.nameplate_cap, 3), ", power deg:", np.round(batt.effective_power/batt.nameplate_power, 3), ", pv deg:", np.round((1-pv_deg)**year, 3)
    
    dispatch_results = dFuncs.determine_optimal_dispatch(load_profile, batt, tariff, export_tariff, d_inc_n, DP_inc, estimator_params=None, estimated=False)
    dispatched_net_profile = dispatch_results['opt_load_profile']
    
    dispatched_bill, dispatched_bill_results = tFuncs.bill_calculator(dispatched_net_profile, tariff, export_tariff)
    
    # no escalation - this will be adjusted depending on the starting year
    ref_bill_savings_cf[year+1] = (original_bill - dispatched_bill)
            

'''
Bring in projections of:
-future electricity cost
-future pv cost
-future battery cost

'''
#%%
scenarios = np.array([['pv_price_mid', 'com_e_price_ref', 'batt_kWh_cost_mid', 'batt_kW_cost_mid'],
                      ['pv_price_low', 'com_e_price_high_oil_price', 'batt_kWh_cost_low', 'batt_kW_cost_low'],
                      ['pv_price_high', 'com_e_price_low_oil_price', 'batt_kWh_cost_high', 'batt_kW_cost_high']])

n_scenarios = np.shape(scenarios)[0]
scenario_names = np.array(['mid', 'high', 'low'])

batt_power_price_2016 = 1600.0
batt_cap_price_2016 = 500.0

years = np.arange(2016, 2022)
results_df = pd.DataFrame(index=years, columns=scenario_names)

for year in years:
    for s in range(n_scenarios):
        batt_power_price = np.array([price_project_df[scenarios[s, 3]][year]])*batt_power_price_2016
        batt_cap_price = np.array([price_project_df[scenarios[s, 2]][year]])*batt_cap_price_2016
        pv_price = np.array([price_project_df[scenarios[s, 0]][year]])
        e_price_esc = np.array([price_project_df[scenarios[s, 1]][np.arange(year, year+analysis_years+1)]])
        bill_savings_cf = ref_bill_savings_cf * e_price_esc
        
        cf_results = fFuncs.cashflow_constructor(bill_savings_cf, 
                                 pv_size, pv_price, inverter_price, pv_om,
                                 batt.nameplate_cap, batt.nameplate_power, batt_power_price, batt_cap_price, batt_chg_frac,
                                 batt_replacement_sch, batt_om,
                                 sector, itc, deprec_sched, 
                                 fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
                                 analysis_years, inflation, 
                                 loan_rate, loan_term, 
                                 cash_incentives, ibi, cbi, pbi)
                                 
        payback = fFuncs.calc_payback_vectorized(cf_results['cf'], analysis_years)
        results_df[scenario_names[s]][year] = payback
        
#%%        
plt.figure(3, figsize=(5,5))
plt.plot(years, results_df) #frac_self_supply_2d
plt.grid(True)
plt.ylabel('Simple payback\nperiod (years)', rotation=0, labelpad=80, size=14)
plt.xlabel('Year', size=14)
plt.title('Change in simple payback over time under future \ntechnology and electricity costs')
plt.axis([2016, 2021, 0, 6.5])
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
            

