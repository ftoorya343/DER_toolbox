# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:59:03 2016

@author: pgagnon
"""

import pandas as pd
import numpy as np
import dispatch_functions as dFuncs
import tariff_functions as tFuncs
import financial_functions as fFuncs
import general_functions as gFuncs

#%%

def system_size_and_bill_calc(agent, e_escalation_sch, deprec_sch, pv_cf_profile, tariff, export_tariff):
    print "starting a sizing..."
    d_inc_n = 20    
    pv_inc = 3
    batt_inc = 3
    load_profile = agent['consumption_hourly']
    pv_sizes = np.linspace(0, agent['max_pv_size'], pv_inc)
    batt_powers = np.linspace(0, np.array(agent['max_demand_kw']) * 0.2, batt_inc)
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
    n_sys = len(system_df)

    for i in system_df.index:    
        pv_size = system_df['pv'][i].copy()
        batt_power = system_df['batt'][i].copy()
        print pv_size, batt_power
        load_and_pv_profile = load_profile - pv_size*pv_cf_profile
        estimator_params = dFuncs.calc_estimator_params(load_and_pv_profile, tariff, export_tariff, batt.eta_charge, batt.eta_discharge)
                
        batt.set_cap_and_power(batt_power*3.0, batt_power)    

        estimated_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimator_params=estimator_params, estimated=True, d_inc_n=d_inc_n)
        system_df.loc[i, 'est_bills'] = estimated_results['bill_under_dispatch']   
        
    est_bill_savings = np.zeros([n_sys, agent['analysis_years']+1])
    est_bill_savings[:,1:] = (original_bill - np.array(system_df['est_bills'])).reshape([n_sys, 1])
    system_df['est_bill_savings'] = (original_bill - np.array(system_df['est_bills'])).reshape([n_sys, 1]) 
    
    batt_chg_frac = 1.0 # just a placeholder...
                
    cf_results_est = fFuncs.cashflow_constructor(est_bill_savings, 
                 system_sizes[:,0], 
                 agent['installed_costs_dollars_per_kw'], 
                 0, #inverter price, assuming it is wrapped into initial and O&M costs
                 agent['fixed_om_dollars_per_kw_per_yr'],
                 system_sizes[:,1]*3,
                 system_sizes[:,1], 
                 agent['batt_cost_per_kW'], 
                 agent['batt_cost_per_kWh'], 
                 batt_chg_frac,
                 agent['batt_replace_yr'],
                 agent['batt_om'],
                 agent['sector'],
                 agent['itc_fraction'],
                 deprec_sch, 
                 agent['tax_rate'], # fed tax rate
                 0, # state tax rate
                 agent['discount_rate'], # real discount
                 agent['down_payment'], 
                 agent['analysis_years'],
                 agent['inflation'], 
                 agent['loan_rate'],
                 agent['loan_term_yrs'])
                                                      
    system_df['npv'] = cf_results_est['npv']
    
    index_of_max_npv = system_df['npv'].idxmax()
    
    opt_pv_size = system_df['pv'][index_of_max_npv].copy()
    opt_batt_power = system_df['batt'][index_of_max_npv].copy()
    batt.set_cap_and_power(opt_batt_power*3.0, opt_batt_power)    
    print "opt pv and batt:", opt_pv_size, opt_batt_power
    load_and_pv_profile = load_profile - opt_pv_size*pv_cf_profile
    accurate_results = dFuncs.determine_optimal_dispatch(load_and_pv_profile, batt, tariff, export_tariff, estimated=False, d_inc_n=d_inc_n)
    opt_bill = accurate_results['bill_under_dispatch']   
    opt_bill_savings = np.zeros([n_sys, agent['analysis_years']+1])
    opt_bill_savings[:,1:] = (original_bill - opt_bill)
    
    cf_results_opt = fFuncs.cashflow_constructor(opt_bill_savings, 
                 opt_pv_size, 
                 agent['installed_costs_dollars_per_kw'], 
                 0, #inverter price, assuming it is wrapped into initial and O&M costs
                 agent['fixed_om_dollars_per_kw_per_yr'],
                 opt_batt_power*3,
                 opt_batt_power, 
                 agent['batt_cost_per_kW'], 
                 agent['batt_cost_per_kWh'], 
                 batt_chg_frac,
                 agent['batt_replace_yr'],
                 agent['batt_om'],
                 agent['sector'],
                 agent['itc_fraction'],
                 deprec_sch, 
                 agent['tax_rate'], # fed tax rate
                 0, # state tax rate
                 agent['discount_rate'], # real discount
                 agent['down_payment'], 
                 agent['analysis_years'],
                 agent['inflation'], 
                 agent['loan_rate'],
                 agent['loan_term_yrs'])   
                 
    results = {"opt_pv_size":opt_pv_size,
               "opt_batt_power":opt_batt_power,
               "npv":cf_results_opt['npv']}
                 
    return system_df, results
    
    
#%%
def system_size_driver(agent, rate_growth_df, deprec_sch_df, pv_cf_profile_df):

    deprec_sch = np.array(deprec_sch_df.loc[agent['deprec_sched_index'], 'deprec'])
    pv_cf_profile = np.array(pv_cf_profile_df.loc[agent['pv_cf_profile_index'], 'generation_hourly'])/1e6
    agent['naep'] = float(np.sum(pv_cf_profile))/1e6 # Is this correct? The 1d6?
    agent['max_pv_size'] = np.min([agent['load_kwh_per_customer_in_bin']/agent['naep'], agent['developable_roof_sqft']*agent['pv_power_density_sqft']*agent['gcr']])
    
    tariff = tFuncs.Tariff(json_file_name='coned_sc9_large_voluntary_tod.json')
    export_tariff = tFuncs.Export_Tariff(full_retail_nem=True)    
    
    sizing_results_df, sizing_results_dict = system_size_and_bill_calc(agent, rate_growth_df, deprec_sch, pv_cf_profile, tariff, export_tariff)

    agent['opt_pv_size'] = sizing_results_dict['opt_pv_size']
    agent['opt_batt_power'] = sizing_results_dict['opt_batt_power']
    
    return agent
    
    
#%%
    
agent_df = pd.read_pickle('interm_pickles/agent_df.pkl')
depreciation_df = pd.read_pickle('interm_pickles/depreciation_df.pkl')
dsire_incentives = pd.read_pickle('interm_pickles/dsire_incentives.pkl')
dsire_opts = pd.read_pickle('interm_pickles/dsire_opts.pkl')
financial_parameters = pd.read_pickle('interm_pickles/financial_parameters.pkl')
incentives_cap = pd.read_pickle('interm_pickles/incentives_cap.pkl')
itc_options = pd.read_pickle('interm_pickles/itc_options.pkl')
normalized_hourly_resource_solar_df = pd.read_pickle('interm_pickles/normalized_hourly_resource_solar_df.pkl')
normalized_load_profiles_df = pd.read_pickle('interm_pickles/normalized_load_profiles_df.pkl')
rates_json_df = pd.read_pickle('interm_pickles/rates_json_df.pkl')
rates_rank_df = pd.read_pickle('interm_pickles/rates_rank_df.pkl')
resource_solar_df = pd.read_pickle('interm_pickles/resource_solar_df.pkl')
srecs = pd.read_pickle('interm_pickles/srecs.pkl')
tech_costs_solar_df = pd.read_pickle('interm_pickles/tech_costs_solar_df.pkl')
rate_growth_df = pd.read_pickle('interm_pickles/rate_growth_df.pkl')

fin_df_ho = financial_parameters[financial_parameters['business_model']=='host_owned']

agent_df = agent_df.merge(fin_df_ho, how='left', on=['year', 'tech', 'sector_abbr'])
agent_df = agent_df.merge(itc_options[['itc_fraction', 'year', 'tech', 'sector_abbr']], how='left', on=['year', 'tech', 'sector_abbr'])
agent_df = agent_df.merge(tech_costs_solar_df[['installed_costs_dollars_per_kw', 'fixed_om_dollars_per_kw_per_yr', 'tech', 'sector_abbr']], how='left', on=['tech', 'sector_abbr'])
agent_df['batt_cost_per_kW'] = 1000.0
agent_df['batt_cost_per_kWh'] = 500.0
agent_df['batt_om'] = 0 # just a placeholder...
agent_df['analysis_years'] = 25
agent_df['batt_replace_yr'] = 10
agent_df['deprec_sched_index'] = int(1)
agent_df['inflation'] = 0.025 # probably would rather not have this included as a column
agent_df['pv_cf_profile_index'] = 0 # placeholder...
agent_df['gcr'] = 0.7
agent_df['pv_power_density_sqft'] = 0.01486 # placeholder... this should be looked up via schedule

agent_df_results = agent_df.apply(system_size_driver, axis=1, args=(rate_growth_df, depreciation_df, normalized_hourly_resource_solar_df))
    
    