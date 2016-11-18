# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:57:34 2016

@author: pgagnon
"""

import sys
sys.path.append('C:/users/pgagnon/desktop/support_functions/python')

import numpy as np
import tariff_functions as tFuncs
import dispatch_functions as dFuncs
import financial_functions as fFuncs
import matplotlib.pyplot as plt

analysis_years = 10
inflation = 0.02
real_d = np.array([0.08])
nom_d = (1 + real_d) * (1 + inflation) - 1

pv_price = np.array([2160.0])
inverter_price = np.array([0.0])
batt_power_price = np.array([1600.0])
batt_cap_price = np.array([500.0])
pv_om = np.array([20.0])
batt_om = np.array([0.0])

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
     
    full_retail_nem = False
    prices = np.zeros([1, 1], float)     
    levels = np.zeros([1, 1], float)
    periods_8760 = np.zeros(8760, int)
    period_tou_n = 1
    
e_escalation = np.zeros(analysis_years+1)
e_escalation[0] = 1.0
e_escalation[1:] = (1.0039)**np.arange(analysis_years)

pv_size = np.array([0.0])
batt_cap = np.array([100.0])
batt_power = batt_cap/2.0
pv_deg = 0.005
    
class batt:
    SOC_min = 0.2    
    power = batt_power[0]
    cap = batt_cap[0]
    cap_effective = cap*(1-SOC_min)
    eta_charge = 0.9107140056021978
    eta_discharge = 0.9107140056021978

#%%    
#tariff_object = tFuncs.Tariff('574e067d5457a349215e629d')
#tariff_object.write_json('coned_sc9_large_voluntary_tod.json')
    
tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_voluntary_tod.json')

profile = np.genfromtxt('input_profile_lg_office_ny.csv', delimiter=",", skip_header=1)
original_load_profile = profile[:,0]
pv_cf_profile = profile[:,1]

pv_profile = pv_size*pv_cf_profile
load_profile = original_load_profile - pv_profile
aep = np.sum(pv_profile)
aec = np.sum(original_load_profile)
energy_penetration = aep / aec
print "annual energy penetration:", energy_penetration

d_inc_n = 50
DP_inc = 50

#%%
def deg_calc(year):
    
    if year < 6.3:
        deg_coeff_cap = (-0.000365*year**3 + 0.006453*year**2 - 0.054878*year + 0.9997)
    else:
        deg_coeff_cap = -0.0301*year + 1.0094

    deg_coeff_power = 1 - (1-deg_coeff_cap)*1.25
    
    return deg_coeff_cap, deg_coeff_power

#%% Only battery deg for just lifetime of battery
n_years = 10
years = range(n_years)

bill_savings = np.zeros(n_years)
demand_savings = np.zeros(n_years)
energy_savings = np.zeros(n_years)
deg_coeff_power = np.zeros(n_years)
deg_coeff_cap = np.zeros(n_years)

batt_starting_power = batt.power
batt_starting_cap = batt.cap

e_escalation = np.ones(analysis_years)

for year in years:
    
    deg_coeff_cap[year], deg_coeff_power[year] = deg_calc(year)
    batt.power = batt_starting_power * deg_coeff_power[year]
    batt.cap = batt_starting_cap * deg_coeff_cap[year]
    batt.cap_effective = batt.cap*(1-batt.SOC_min)
    
    print year, "cap deg:", deg_coeff_cap[year], ", power deg:", deg_coeff_power[year]
    
    dispatch_results = dFuncs.determine_optimal_dispatch(load_profile, batt, tariff, export_tariff, d_inc_n, DP_inc)
    dispatched_net_profile = dispatch_results['opt_load_profile']
    
    original_bill, original_bill_results = tFuncs.bill_calculator(original_load_profile, tariff, export_tariff)
    dispatched_bill, dispatched_bill_results = tFuncs.bill_calculator(dispatched_net_profile, tariff, export_tariff)
    
    original_bill = original_bill *e_escalation[year]
    dispatched_bill = dispatched_bill*e_escalation[year]
    
    bill_savings[year] = original_bill - dispatched_bill
    demand_savings[year] = original_bill_results['d_charges'] - dispatched_bill_results['d_charges']
    energy_savings[year] = original_bill_results['e_charges'] - dispatched_bill_results['e_charges']
    

#bill_savings = np.insert(bill_savings, 0, 0)

#cf_results = fFuncs.cashflow_constructor(bill_savings, 
#                         pv_size, pv_price, inverter_price, pv_om,
#                         batt_cap, batt_power, batt_power_price, batt_cap_price, batt_chg_frac,
#                         batt_replacement_sch, batt_om,
#                         sector, itc, deprec_sched, 
#                         fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
#                         analysis_years, inflation, 
#                         loan_rate, loan_term, 
#                         cash_incentives, ibi, cbi, pbi)
                     
                     
#%%
plt.plot(years, bill_savings/max(bill_savings))
plt.plot(years, deg_coeff_cap/max(deg_coeff_cap))
plt.legend(['bill savings', 'degradation'])



#%% Deg with both pv and battery with replacement
if False:
    n_years = 20
    years = range(n_years)
    
    bill_savings = np.zeros(n_years)
    demand_savings = np.zeros(n_years)
    energy_savings = np.zeros(n_years)
    
    batt_starting_power = batt.power
    batt_starting_cap = batt.cap
    
    e_escalation = np.ones(analysis_years)
    
    for year in years:
        
        deg_year = np.mod(year,10)
        deg_coeff_cap = (-0.000365*deg_year**3 + 0.006453*deg_year**2 - 0.054878*deg_year + 0.9997)
        deg_coeff_power = 1 - (1-deg_coeff_cap)*1.25
        batt.power = batt_starting_power * deg_coeff_power
        batt.cap = batt_starting_cap * deg_coeff_cap
        batt.cap_effective = batt.cap*(1-batt.SOC_min)
        #pv_profile = pv_size*pv_cf_profile*(1-pv_deg)**year
        #load_profile = original_load_profile - pv_profile
        
        print year, "cap deg:", deg_coeff_cap, ", power deg:", deg_coeff_power, ", pv deg:", (1-pv_deg)**year
    
        
        dispatch_results = dFuncs.determine_optimal_dispatch(load_profile, batt, tariff, export_tariff, d_inc_n, DP_inc)
        dispatched_net_profile = dispatch_results['opt_load_profile']
        
        original_bill, original_bill_results = tFuncs.bill_calculator(original_load_profile, tariff, export_tariff)
        dispatched_bill, dispatched_bill_results = tFuncs.bill_calculator(dispatched_net_profile, tariff, export_tariff)
        
        original_bill = original_bill *e_escalation[year]
        dispatched_bill = dispatched_bill*e_escalation[year]
        
        bill_savings[year] = original_bill - dispatched_bill
        fy_demand_savings = original_bill_results['d_charges'] - dispatched_bill_results['d_charges']
        fy_energy_savings = original_bill_results['e_charges'] - dispatched_bill_results['e_charges']
        
    
    bill_savings = np.insert(bill_savings, 0, 0)
    
    cf_results = fFuncs.cashflow_constructor(bill_savings, 
                             pv_size, pv_price, inverter_price, pv_om,
                             batt_cap, batt_power, batt_power_price, batt_cap_price, batt_chg_frac,
                             batt_replacement_sch, batt_om,
                             sector, itc, deprec_sched, 
                             fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
                             analysis_years, inflation, 
                             loan_rate, loan_term, 
                             cash_incentives, ibi, cbi, pbi)
                             
    #npv = fFuncs.calc_npv(cf_results['cf'], nom_d)
    #npv_bill_savings = fFuncs.calc_npv(cf_results['after_tax_bill_savings'], nom_d)
    #npv_d_savings = npv_bill_savings*savings_frac_d
    #npv_e_savings = npv_bill_savings*savings_frac_e
    #npv_batt_replace = fFuncs.calc_npv(cf_results['batt_replacement'], nom_d)
    #npv_operating = fFuncs.calc_npv(cf_results['operating_expenses'], nom_d)
    #itc_cf = np.zeros([1,26])
    #itc_cf[:,1] = cf_results['itc_value']
    #npv_itc = fFuncs.calc_npv(itc_cf, nom_d)
    #npv_state_income_tax = fFuncs.calc_npv(cf_results['state_income_taxes'], nom_d)
    #npv_fed_income_tax = fFuncs.calc_npv(cf_results['fed_income_taxes'], nom_d)
    
    
    
