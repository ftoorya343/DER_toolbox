# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:37:22 2016

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
inflation_adjustment = (1+inflation)**np.arange(analysis_years+1)

hottest_day_index = 4069 #June 19th, 11:30AM

e_escalation = np.zeros(analysis_years+1)
e_escalation[0] = 1.0
e_escalation[1:] = (1.0039)**np.arange(analysis_years)

pv_size = np.array([500.0])
batt_cap = np.array([500.0])
batt_power = np.array([250.0])

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
bill_savings_cf = np.zeros(analysis_years+1)
demand_savings_cf = np.zeros(analysis_years+1)
energy_savings_cf = np.zeros(analysis_years+1)

for year in years:
    batt.set_cycle_deg(365*5/7*np.mod(year,10)) #Assume it cycles 5 times per week
    
    pv_profile = pv_size*pv_cf_profile*(1-pv_deg)**year
    load_profile = original_load_profile - pv_profile
    
    print year, "cap deg:", np.round(batt.effective_cap/batt.nameplate_cap, 3), ", power deg:", np.round(batt.effective_power/batt.nameplate_power, 3), ", pv deg:", np.round((1-pv_deg)**year, 3)
    
    dispatch_results = dFuncs.determine_optimal_dispatch(load_profile, batt, tariff, export_tariff, d_inc_n, DP_inc, estimator_params=None, estimated=False)
    dispatched_net_profile = dispatch_results['opt_load_profile']
    
    dispatched_bill, dispatched_bill_results = tFuncs.bill_calculator(dispatched_net_profile, tariff, export_tariff)
    
    bill_savings_cf[year+1] = (original_bill - dispatched_bill) * e_escalation[year]
            
    demand_savings_cf[year+1] = (original_bill_results['d_charges'] - dispatched_bill_results['d_charges']) * e_escalation[year] * inflation_adjustment[year+1]
    energy_savings_cf[year+1] = (original_bill_results['e_charges'] - dispatched_bill_results['e_charges']) * e_escalation[year] * inflation_adjustment[year+1]

cf_results = fFuncs.cashflow_constructor(bill_savings_cf, 
                             pv_size, pv_price, inverter_price, pv_om,
                             batt.nameplate_cap, batt.nameplate_power, batt_power_price, batt_cap_price, batt_chg_frac,
                             batt_replacement_sch, batt_om,
                             sector, itc, deprec_sched, 
                             fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
                             analysis_years, inflation, 
                             loan_rate, loan_term, 
                             cash_incentives, ibi, cbi, pbi)
                             
                             
#%%
npv = fFuncs.calc_npv(cf_results['cf'], nom_d)
npv_bill_savings = fFuncs.calc_npv(np.array([cf_results['bill_savings']]), nom_d)
npv_d_savings = fFuncs.calc_npv(np.array([demand_savings_cf]), nom_d)
npv_e_savings = fFuncs.calc_npv(np.array([energy_savings_cf]), nom_d)
npv_batt_replace = fFuncs.calc_npv(cf_results['batt_replacement'], nom_d)
npv_operating = fFuncs.calc_npv(cf_results['operating_expenses'], nom_d)
itc_cf = np.zeros([1,26])
itc_cf[:,1] = cf_results['itc_value']
npv_itc = fFuncs.calc_npv(itc_cf, nom_d)
npv_state_income_tax = fFuncs.calc_npv(cf_results['state_income_taxes'], nom_d)
npv_fed_income_tax = fFuncs.calc_npv(cf_results['fed_income_taxes'], nom_d)

npv_OM_tax_savings = fFuncs.calc_npv(cf_results['operating_expenses_tax_savings'], nom_d)
npv_deprec_tax_savings = fFuncs.calc_npv(cf_results['deprec_deductions_tax_savings'], nom_d)
npv_elec_OM_tax_liability = fFuncs.calc_npv(cf_results['elec_OM_deduction_decrease_tax_liability'], nom_d)



#%%
print "\n"
print "total npv: $", npv[0]
print "installed cost: $", cf_results['installed_cost'][0]
print "bill savings: $", npv_bill_savings[0]
print "d savings: $", npv_d_savings[0]
print "e savings: $", npv_e_savings[0]
print "batt replace: $", npv_batt_replace[0]
print "operating: $", npv_operating[0]
print "itc: $", npv_itc[0]
print "state tax: $", npv_state_income_tax[0]
print "fed tax: $", npv_fed_income_tax[0]


#%%
pv_cost_ratio = cf_results['pv_cost'][0] / cf_results['installed_cost'][0]
batt_cost_ratio = cf_results['batt_cost'][0] / cf_results['installed_cost'][0]

cost_stack = (cf_results['pv_cost'][0], cf_results['batt_cost'][0], -npv_batt_replace[0], npv_operating[0], npv_elec_OM_tax_liability[0])
cost_stack_names = ('PV cost', 'Battery cost', 'battery replacement', 'Operating costs', 'Increased tax liability\nfrom reduced elec O&M')

revenue_stack = (npv_d_savings[0], npv_e_savings[0], npv_itc[0], npv_OM_tax_savings[0], npv_deprec_tax_savings[0])
revenue_stack_names = ('demand charge savings', 'energy savings', 'ITC', 'O&M income tax deductions', 'depreciation deductions')

stacks = np.array([cost_stack, revenue_stack])
stack_names = np.array([cost_stack_names, revenue_stack_names])

def stacked_bar_chart(stacks, stack_names):
    colors = ['gold', 'orangered', 'mediumseagreen', 'cornflowerblue', 'darkseagreen', 'darkorange']
    width = 0.8
    for s in range(len(stacks)):
        for c in range(len(stacks[s])):        
            #fig = plt.figure(1)
            start = np.sum(stacks[s][0:c])
            plt.bar(s, stacks[s][c], bottom=start, color=colors[c], width=width)
            plt.text(s+.1, start+stacks[s][c]*0.5, stack_names[s][c])
        

stacked_bar_chart(stacks, stack_names)
