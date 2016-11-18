# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:53:47 2016

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

pv_price = np.array([2160.0])
batt_power_price = np.array([1600.0])
batt_cap_price = np.array([500.0])

inverter_price = np.array([0.0])
pv_om = np.array([20.0])
batt_om = np.array([0.0])

pv_deg = 0.005

batt_replacement_sch = np.array([10])
sector = np.array(['com'])

itc = np.array([0.3])
deprec_sched = np.array([0.6, .16, .096, 0.0576, 0.0576, .0288])
#deprec_sched = np.zeros([1,len(deprec_sched_single)]) + deprec_sched_single

fed_tax_rate = np.array([0.35])
state_tax_rate = np.array([0.0])

debt_fraction = np.array([0.0])
loan_rate = np.array([0.05])
loan_term = np.array(20)

cash_incentives = np.array([0])
ibi = np.array([0])
cbi = np.array([0])
pbi = np.array([0])
    
#%% Import commercial willingess-to-pay curves  
wtp = np.genfromtxt('com_wtp.csv', delimiter=",", skip_header=1)
wtp_df = pd.read_csv('com_wtp.csv')
#%%    
#tariff_object = tFuncs.Tariff('574e067d5457a349215e629d')
#tariff_object.write_json('coned_sc9_large_voluntary_tod.json')
tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_tod.json')

full_retail_nem = True
export_tariff = tFuncs.Export_Tariff(full_retail_nem)

#%% Import profiles
load_profiles = pd.read_csv('bld_load_profiles.csv')
pv_cf_profile = np.genfromtxt('pv_cf_profile.csv', delimiter=",", skip_header=1)
naep = np.sum(pv_cf_profile)


#%% Define building characters

blds = ['full_service_restaurant', 'hospital', 'large_hotel', 'large_office', 'medium_office', 'midrise_apartment', 'outpatient', 'primary_school', 'quick_service_restaurant', 'secondary_school', 'small_hotel', 'small_office', 'standalone_retail', 'strip_mall', 'supermarket', 'warehouse']
bld_df = pd.DataFrame(index=blds)
bld_df['max_pv_kw'] = [26, 296, 96, 255, 84, 40, 64, 454, 12, 647, 51, 26, 118, 106, 276, 320]
bld_df['peak_kw'] = None
bld_df['aec'] = None
bld_df['tariff_category'] = 'demand'
bld_df['estimated_fraction'] = None
demand_metered_n = 127000
interval_metered_n = 2500


# This is just a placeholder. There is also middle atlantic data. Plus,
# I probably want to do some basic validation with total floorspace, kW peak, etc...
bld_fractions_of_nat_pop = {'full_service_restaurant':0.088578,
                            'hospital':0.002331, 
                            'large_hotel':0.013995, 
                            'large_office':0.075487, 
                            'medium_office':0.096718, 
                            'midrise_apartment':0, 
                            'outpatient':0.034266, 
                            'primary_school':0.045338, 
                            'quick_service_restaurant':0.144289, 
                            'secondary_school':0.045338, 
                            'small_hotel':0.022834, 
                            'small_office':0.063692, 
                            'standalone_retail':0.102098, 
                            'strip_mall':0.038228, 
                            'supermarket':0.041259, 
                            'warehouse':0.185548}

for bld in blds:
    bld_df.loc[bld, 'peak_kw'] = np.max(load_profiles[bld])    
    bld_df.loc[bld, 'aec'] = np.sum(load_profiles[bld]) 
    bld_df.loc[bld, 'estimated_fraction'] = bld_fractions_of_nat_pop[bld]   
    if bld=='large_office' or bld=='hospital' or bld=='large_hotel' or bld=='secondary_school': bld_df.loc[bld, 'tariff_category'] = 'interval' 

demand_weight = 0
interval_weight = 0
bld_df['category_fraction'] = None
for bld in blds:
    if bld_df.loc[bld, 'tariff_category'] == 'demand': demand_weight += bld_df.loc[bld, 'estimated_fraction']
    if bld_df.loc[bld, 'tariff_category'] == 'interval': interval_weight += bld_df.loc[bld, 'estimated_fraction']
        
for bld in blds:
    if bld_df.loc[bld, 'tariff_category'] == 'demand': 
        bld_df.loc[bld, 'category_fraction'] = bld_df.loc[bld, 'estimated_fraction'] / demand_weight
        bld_df.loc[bld, 'n_blds'] = bld_df.loc[bld, 'category_fraction'] * demand_metered_n
    if bld_df.loc[bld, 'tariff_category'] == 'interval': 
        bld_df.loc[bld, 'category_fraction'] = bld_df.loc[bld, 'estimated_fraction'] / interval_weight
        bld_df.loc[bld, 'n_blds'] = bld_df.loc[bld, 'category_fraction'] * interval_metered_n


#%%
d_inc_n = 50
DP_inc = 50
years = range(analysis_years)

incentive_inc = 50 
max_incentive = 1.0
incentive_levels = np.linspace(0, max_incentive, incentive_inc)

paybacks_df = pd.DataFrame()
frac_wtp_df = pd.DataFrame()
n_wtp_df = pd.DataFrame()
pv_kw_wtp_df = pd.DataFrame()
batt_kw_wtp_df = pd.DataFrame()
batt_kwh_wtp_df = pd.DataFrame()
inc_payout_df = pd.DataFrame()
    
paybacks_df['inc_frac'] = incentive_levels
paybacks_df['inc_frac'] = incentive_levels
frac_wtp_df['inc_frac'] = incentive_levels
n_wtp_df['inc_frac'] = incentive_levels
pv_kw_wtp_df['inc_frac'] = incentive_levels
batt_kw_wtp_df['inc_frac'] = incentive_levels
batt_kwh_wtp_df['inc_frac'] = incentive_levels
inc_payout_df['inc_frac'] = incentive_levels
    
for bld in blds:
    print bld
    base_load_profile = np.array(load_profiles[bld])
    pv_size = bld_df.loc[bld, 'max_pv_kw']
    batt_kw = pv_size * 0.25
    batt_kwh = batt_kw * 2.0
    
    batt = dFuncs.Battery(nameplate_cap=batt_kwh, nameplate_power=batt_kw)
    
    pv_profile = pv_size*pv_cf_profile
    aep = np.sum(pv_profile)
    bld_df.loc[bld, 'energy_penetration'] = aep / bld_df.loc[bld, 'aec']
    
    original_bill, original_bill_results = tFuncs.bill_calculator(base_load_profile, tariff, export_tariff)
    bill_savings_cf = np.zeros(analysis_years+1)
    
    for year in years:
        batt.set_cycle_deg(365*5/7*np.mod(year,10)) #Assume it cycles 5 times per week
        batt_chg_frac = np.array([1.0])
    
        pv_profile = pv_size*pv_cf_profile*(1-pv_deg)**year
        load_profile = base_load_profile - pv_profile
        
        print year, "cap deg:", np.round(batt.effective_cap/batt.nameplate_cap, 3), ", power deg:", np.round(batt.effective_power/batt.nameplate_power, 3), ", pv deg:", np.round((1-pv_deg)**year, 3)
        
        dispatch_results = dFuncs.determine_optimal_dispatch(load_profile, batt, tariff, export_tariff, d_inc_n, DP_inc, estimated=False)
        dispatched_net_profile = dispatch_results['opt_load_profile']
        
        dispatched_bill, dispatched_bill_results = tFuncs.bill_calculator(dispatched_net_profile, tariff, export_tariff)
        
        bill_savings_cf[year+1] = (original_bill - dispatched_bill) * e_escalation[year+1]
                
    
    
    bill_savings_cfs = np.zeros([incentive_inc, analysis_years+1])
    bill_savings_cfs[:,:] = bill_savings_cf
    cbis = incentive_levels  * (batt.nameplate_cap*batt_cap_price + batt.nameplate_power*batt_power_price)
    
    cf_results = fFuncs.cashflow_constructor(bill_savings_cfs, 
                                 pv_size, pv_price, inverter_price, pv_om,
                                 batt.nameplate_cap, batt.nameplate_power, batt_power_price, batt_cap_price, batt_chg_frac,
                                 batt_replacement_sch, batt_om,
                                 sector, itc, deprec_sched, 
                                 fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
                                 analysis_years, inflation, 
                                 loan_rate, loan_term, 
                                 cash_incentives, ibi, cbis, pbi)
                                 
    paybacks_df[bld] = fFuncs.calc_payback_vectorized(cf_results['cf'], analysis_years)
    frac_wtp_df[bld] = np.interp(paybacks_df[bld], wtp_df['simple_payback_period'], wtp_df['fraction_willing_to_adopt'])
    n_wtp_df[bld] = bld_df.loc[bld, 'n_blds'] * frac_wtp_df[bld]
    pv_kw_wtp_df[bld] = n_wtp_df[bld] * pv_size
    batt_kw_wtp_df[bld] = n_wtp_df[bld] * batt.nameplate_power
    batt_kwh_wtp_df[bld] = n_wtp_df[bld] * batt.nameplate_cap
    inc_payout_df[bld] = n_wtp_df[bld] * cbis
    


#%% 
plt.figure(1, figsize=(5,3))
plt.plot(incentive_levels, paybacks_df)
plt.legend(paybacks_df.columns)
plt.grid(True)
plt.ylabel('Simple payback\nperiod (years)', rotation=0, labelpad=80, size=14)
plt.xlabel('Incentive level (fraction of total battery cost)', size=14)
plt.title('Impact of incentive level on simple payback period')
#plt.axis([0, max_incentive, 0, 6.5])

plt.figure(2, figsize=(5,3))
plt.plot(incentive_levels, frac_wtp_df)
plt.grid(True)
plt.ylabel('Fraction of potential\ncommercial customers\nwilling to adopt', rotation=0, labelpad=80, size=14)
plt.xlabel('Incentive level (fraction of total battery cost)', size=14)
plt.title('Impact of incentive level on the fraction\nof commercial customers willing to adopt')
plt.axis([0, max_incentive, 0, 1])

plt.figure(3, figsize=(5,3))
plt.plot(incentive_levels, pv_kw_wtp_df/1000, incentive_levels, batt_kw_wtp_df/1000)
plt.grid(True)
plt.ylabel('Amount of capacity\nthat customers would be\nwilling to adopt (MW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Incentive level (fraction of total battery cost)', size=14)
plt.title('Impact of incentive level on the capacity\nof commercial customers willing to adopt')
#plt.axis([0, max_incentive, 0, 1])

plt.figure(4, figsize=(5,3))
plt.plot(incentive_levels, np.sum(pv_kw_wtp_df/1000, 1), incentive_levels, np.sum(batt_kw_wtp_df/1000, 1))
plt.grid(True)
plt.ylabel('Amount of capacity\nthat customers would be\nwilling to adopt (MW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Incentive level (fraction of total battery cost)', size=14)
plt.title('Impact of incentive level on the capacity\nof commercial customers willing to adopt')