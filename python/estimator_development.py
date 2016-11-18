# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:50:52 2016

@author: pgagnon
"""

import numpy as np
import pandas as pd
import tariff_functions as tFuncs
import dispatch_functions as dFuncs
import financial_functions as fFuncs


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


#%% Create class objects        
export_tariff = tFuncs.Export_Tariff(full_retail_nem=True)
 
batt = dFuncs.Battery()
    
#tariff = tFuncs.Tariff(urdb_id='574e067d5457a349215e629d') #not a real rate: 57d0b2315457a3120ec5b286
#tariff.write_json('coned_sc9_large_voluntary_tod.json')
tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_voluntary_tod.json')
tar = tariff.__dict__


#%% Import profiles
profiles = pd.read_csv('input_profile_lg_office_ny.csv')
load_profile = np.array(profiles['load'])
pv_cf_profile = np.array(profiles['pv_cf'])


#%% Single
pv_size = 0
batt_power = 10
original_bill, original_results = tFuncs.bill_calculator(load_profile, tariff, export_tariff)
org_period_sums = original_results['period_e_sums']
batt.set_cap_and_power(batt_power*3.0, batt_power)  

load_and_pv_profile = load_profile - pv_size*pv_cf_profile
estimator_params = dFuncs.calc_estimator_params(load_and_pv_profile, tariff, export_tariff, batt.eta_charge, batt.eta_discharge)  

estimated_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimator_params=estimator_params, estimated=True)
estimated_bill = estimated_results['bill_under_dispatch']        
        
accurate_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimated=False)
accurate_bill = accurate_results['bill_under_dispatch']
dispatched_profile = accurate_results['opt_load_profile']
accurate_bill2, accurate_bill_results = tFuncs.bill_calculator(dispatched_profile, tariff, export_tariff)
acc_period_sums = accurate_bill_results['period_e_sums']

estimated_savings = original_bill - estimated_bill
accurate_savings = original_bill - accurate_bill
error_fraction = (accurate_savings-estimated_savings) / accurate_savings


#%% Array
if True:
    pv_inc = 8
    batt_inc = 8
    pv_sizes = np.linspace(0, 2000.0, pv_inc)
    batt_powers = np.linspace(0, 1000.0, batt_inc)
    estimated_bills = np.zeros([pv_inc, batt_inc], float)
    accurate_bills = np.zeros([pv_inc, batt_inc], float)
    paybacks_est = np.zeros([pv_inc, batt_inc], float)
    paybacks_acc = np.zeros([pv_inc, batt_inc], float)
    npvs_est = np.zeros([pv_inc, batt_inc], float)
    npvs_acc = np.zeros([pv_inc, batt_inc], float)
    original_bill, original_results = tFuncs.bill_calculator(load_profile, tariff, export_tariff)
    
    for p, pv_size in enumerate(pv_sizes):
        load_and_pv_profile = load_profile - pv_size*pv_cf_profile
        estimator_params = dFuncs.calc_estimator_params(load_and_pv_profile, tariff, export_tariff, batt.eta_charge, batt.eta_discharge)
        for b, batt_power in enumerate(batt_powers):
                
            batt.set_cap_and_power(batt_power*3.0, batt_power)    
    
            estimated_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimator_params=estimator_params, estimated=True, d_inc_n=20)
            estimated_bills[p,b] = estimated_results['bill_under_dispatch']       
            est_bill_savings = np.zeros(analysis_years+1)
            est_bill_savings[1:] = original_bill - estimated_results['bill_under_dispatch'] 
                    
            accurate_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimated=False)
            accurate_bills[p,b] = accurate_results['bill_under_dispatch']
            acc_bill_savings = np.zeros(analysis_years+1)
            acc_bill_savings[1:] = original_bill - accurate_results['bill_under_dispatch'] 
            
            cf_results_est = fFuncs.cashflow_constructor(est_bill_savings, 
                         pv_size, pv_price, inverter_price, pv_om,
                         batt.nameplate_cap, batt.nameplate_power, batt_power_price, batt_cap_price, batt_chg_frac,
                         batt_replacement_sch, batt_om,
                         sector, itc, deprec_sched, 
                         fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
                         analysis_years, inflation, 
                         loan_rate, loan_term)
                         
            cf_results_acc = fFuncs.cashflow_constructor(acc_bill_savings, 
                         pv_size, pv_price, inverter_price, pv_om,
                         batt.nameplate_cap, batt.nameplate_power, batt_power_price, batt_cap_price, batt_chg_frac,
                         batt_replacement_sch, batt_om,
                         sector, itc, deprec_sched, 
                         fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
                         analysis_years, inflation, 
                         loan_rate, loan_term)
                         
            paybacks_est[p,b] = fFuncs.calc_payback_vectorized(cf_results_est['cf'], analysis_years)
            paybacks_acc[p,b] = fFuncs.calc_payback_vectorized(cf_results_acc['cf'], analysis_years)

            npvs_est[p,b] = cf_results_est['npv']
            npvs_acc[p,b] = cf_results_acc['npv']
            
            print p, b, npvs_est[p,b], npvs_acc[p,b], paybacks_est[p,b], paybacks_acc[p,b]

    #%%
    estimated_bill_savings = (original_bill-estimated_bills)
    accurate_bill_savings = (original_bill-accurate_bills)
    err_savings = (estimated_bill_savings - accurate_bill_savings) / accurate_bill_savings
    
#%%
err_npvs = (npvs_est-npvs_acc) / npvs_acc
err_npvs_norm = (npvs_est-npvs_acc) / np.max(npvs_acc)
err_paybacks = (paybacks_est-paybacks_acc) / paybacks_acc
