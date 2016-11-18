# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:38:32 2016

@author: pgagnon
"""

import pandas as pd
import numpy as np
import dispatch_functions as dFuncs
import tariff_functions as tFuncs
import financial_functions as fFuncs
import general_functions as gFuncs


# global inputs
analysis_years = 20
inflation = 0.02



# agent inputs
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

#%% Build agent_df

e_escalation_df = pd.DataFrame(e_escalation)
deprec_sched_df = pd.DataFrame(deprec_sched)
load_profile_df = pd.DataFrame(load_profile)
pv_cf_profile_df = pd.DataFrame(pv_cf_profile)

agent_df = pd.DataFrame()
agent_df['real_d'] = real_d
agent_df['nom_d'] = nom_d
agent_df['e_escalation_index'] = 0
agent_df['pv_price'] = pv_price
agent_df['batt_power_price'] = batt_power_price
agent_df['batt_cap_price'] = batt_cap_price
agent_df['pv_om'] = pv_om
agent_df['batt_om'] = batt_om
agent_df['pv_deg'] = pv_deg
agent_df['batt_replacement_sch'] = batt_replacement_sch
agent_df['sector'] = sector
agent_df['itc'] = itc
agent_df['deprec_sched_index'] = 0
agent_df['fed_tax_rate'] = fed_tax_rate
agent_df['state_tax_rate'] = state_tax_rate
agent_df['debt_fraction'] = debt_fraction
agent_df['loan_rate'] = loan_rate
agent_df['loan_term'] = loan_term
agent_df['cash_incentives'] = cash_incentives
agent_df['ibi'] = ibi
agent_df['cbi'] = cbi
agent_df['pbi'] = pbi
agent_df['load_profile_index'] = 0
agent_df['pv_cf_profile_index'] = 0
agent_df['tariff_id'] = 'coned_sc9_large_voluntary_tod.json'
agent_df['export_tariff_id'] = True
agent_df['naep'] = np.sum(pv_cf_profile)
agent_df['aec'] = np.sum(load_profile) 
agent_df['max_demand'] = np.max(load_profile)
agent_df['suitable_roof_area_m2'] = 250.0
agent_df['gcr'] = 0.7
agent_df['pv_power_density'] = 0.2

#%%

def system_size_and_bill_calc(agent, e_escalation_df, deprec_sched_df, load_profile_df, pv_cf_profile_df):
    d_inc_n = 20    
    pv_inc = 3
    batt_inc = 3
    load_profile = np.array(load_profile_df[agent['load_profile_index']])[:,0]
    pv_cf_profile = np.array(pv_cf_profile_df[agent['pv_cf_profile_index']])[:,0]
    max_pv_size = np.min([agent['aec']/agent['naep'], agent['suitable_roof_area_m2']*agent['pv_power_density']*agent['gcr']])
    pv_sizes = np.linspace(0, max_pv_size, pv_inc)
    batt_powers = np.linspace(0, np.array(agent['max_demand']) / 2.0, batt_inc)
    tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_voluntary_tod.json')
    export_tariff = tFuncs.Export_Tariff(full_retail_nem=True)
    original_bill, original_results = tFuncs.bill_calculator(load_profile, tariff, export_tariff)
    batt = dFuncs.Battery()
    
#    params_df = pd.DataFrame(index=pv_sizes, columns = ['params'], dtype=object)
#
#    for p, pv_size in enumerate(pv_sizes):
#        load_and_pv_profile = load_profile - pv_size*pv_cf_profile
#        estimator_params = dFuncs.calc_estimator_params(load_and_pv_profile, tariff, export_tariff, batt.eta_charge, batt.eta_discharge)
#        params_df['params'][pv_size] = estimator_params
    
    system_sizes = gFuncs.cartesian([pv_sizes, batt_powers])
    
    system_df = pd.DataFrame(system_sizes, columns=['pv', 'batt'])
    system_df['est_bills'] = None
    system_df['est_bill_savings'] = None
    n_sys = len(system_df)

    for i in system_df.index:    
        pv_size = system_df['pv'][i].copy()
        batt_power = system_df['batt'][i].copy()
        print pv_size, batt_power
        load_and_pv_profile = load_profile - pv_size*pv_cf_profile
        estimator_params = dFuncs.calc_estimator_params(load_and_pv_profile, tariff, export_tariff, batt.eta_charge, batt.eta_discharge)
                
        batt.set_cap_and_power(batt_power*3.0, batt_power)    

        estimated_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimator_params=estimator_params, estimated=True, d_inc_n=d_inc_n)
        system_df.loc[i, 'est_bills'] =   estimated_results['bill_under_dispatch']   
        
    est_bill_savings = np.zeros([n_sys, analysis_years+1])
    est_bill_savings[:,1:] = (original_bill - np.array(system_df['est_bills'])).reshape([n_sys, 1])
    system_df['est_bill_savings'] = (original_bill - np.array(system_df['est_bills'])).reshape([n_sys, 1]) 
                
    cf_results_est = fFuncs.cashflow_constructor(est_bill_savings, 
                 system_sizes[:,0], pv_price, inverter_price, pv_om,
                 system_sizes[:,1]*3, system_sizes[:,1], 
                 batt_power_price, batt_cap_price, batt_chg_frac,
                 batt_replacement_sch,
                 batt_om,
                 sector,
                 itc,
                 deprec_sched, 
                 fed_tax_rate,
                 state_tax_rate,
                 real_d,
                 debt_fraction, 
                 analysis_years,
                 inflation, 
                 loan_rate,
                 loan_term)
                                                      
    system_df['npv'] = cf_results_est['npv']
    
    index_of_max_npv = system_df['npv'].idxmax()
    
    opt_pv_size = system_df['pv'][index_of_max_npv].copy()
    opt_batt_power = system_df['batt'][index_of_max_npv].copy()
    batt.set_cap_and_power(opt_batt_power*3.0, opt_batt_power)    
    print "max pv and batt:", opt_pv_size, opt_batt_power
    load_and_pv_profile = load_profile - opt_pv_size*pv_cf_profile
    accurate_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimated=False, d_inc_n=d_inc_n)
    opt_bill = accurate_results['bill_under_dispatch']   
    opt_bill_savings = np.zeros([n_sys, analysis_years+1])
    opt_bill_savings[:,1:] = (original_bill - opt_bill)
    
    cf_results_opt = fFuncs.cashflow_constructor(opt_bill_savings, 
                 opt_pv_size, pv_price, inverter_price, pv_om,
                 opt_batt_power*3, opt_batt_power, 
                 batt_power_price, batt_cap_price, batt_chg_frac,
                 batt_replacement_sch,
                 batt_om,
                 sector,
                 itc,
                 deprec_sched, 
                 fed_tax_rate,
                 state_tax_rate,
                 real_d,
                 debt_fraction, 
                 analysis_years,
                 inflation, 
                 loan_rate,
                 loan_term)    
                 
    results = {"opt_pv_size":opt_pv_size,
               "opt_batt_power":opt_batt_power,
               "npv":cf_results_opt['npv']}
                 
    return system_df, results
    
    
results_df, results_dict = system_size_and_bill_calc(agent_df, e_escalation_df, deprec_sched_df, load_profile_df, pv_cf_profile_df)