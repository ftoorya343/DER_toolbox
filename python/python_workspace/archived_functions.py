# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:11:36 2016

@author: pgagnon
"""

#%% Energy Arbitrage Value Estimator
def calc_estimator_params(load_and_pv_profile, tariff, NEM_tariff, eta_charge, eta_discharge):
    '''
    This was the first estimator, and is retired because the approach didn't
    work with arbitary export tariffs, just full-retail NEM.    
    
    
    This function creates two 12x24 matrixes that are the marginal value of 
    energy (for each hour of each month) for the given load profile on the 
    given rate.
    
    Assumptions:
        -TOU windows are aligned with when the battery would be dispatching for
         demand peak shaving.
        -The battery will be able to dispatch fully and recharge fully every 24
         hour cycle.
    
    To Do:
        -Consider coming up with a better method that captures exportation, CPP, etc
         Maybe? Or just confirm a simple estimation works with our dGen set, 
         and use the accurate dispatch for any other analysis.
    
    '''
  
    #load_and_pv_profile = df_row['consumption_hourly'] - df_row['generation_hourly']
        
    # Calculate the energy charges for the input load profile    
    annual_bill, tariff_results = tFuncs.bill_calculator(load_and_pv_profile, tariff, NEM_tariff)
    e_period_sums = tariff_results['period_e_sums']
    e_charges = tariff_results['period_e_charges']
    
    # Add 1 kW to every hour of the load profile and recalculate the new energy charges
    load_and_pv_profile_plus_1kw = load_and_pv_profile + 1
    annual_bill, tariff_results2 = tFuncs.bill_calculator(load_and_pv_profile_plus_1kw, tariff, NEM_tariff)
    e_period_sums2 = tariff_results2['period_e_sums']
    e_charges2 = tariff_results2['period_e_charges']
    
    # Determine the marginal energy savings in $/kWh for each of the periods
    marg_consum = e_period_sums2 - e_period_sums
    marg_e_charges = e_charges2 - e_charges
    
    marg_e_chrg_per_kWh = marg_e_charges / marg_consum
    marg_e_chrg_per_kWh = marg_e_chrg_per_kWh.reshape(12,int(len(marg_e_chrg_per_kWh)/12))
    
    # Extract the 12x24 energy period schedules and adjust by 1 so it is an index
    wkday_schedule = tariff.e_sched_weekday - 1
    wkend_schedule = tariff.e_sched_weekend - 1
    
    # Create two 12x24's of the value of energy in each hour of each month
    e_wkday_value_schedule = np.zeros([12,24])
    e_wkend_value_schedule = np.zeros([12,24])
    for row in range(12):
        e_wkday_value_schedule[row,:] = marg_e_chrg_per_kWh[row,wkday_schedule[row,:]]
        e_wkend_value_schedule[row,:] = marg_e_chrg_per_kWh[row,wkend_schedule[row,:]]
    
    # Sort the matrix such that the each month has increasing marginal energy savings 
    sorted_e_wkday_value = np.sort(e_wkday_value_schedule)
    sorted_e_wkend_value = np.sort(e_wkend_value_schedule)

    results = {'e_chrgs_with_PV':sum(tariff_results['period_e_charges']),
                'arbitrage_value_wkday':sorted_e_wkday_value,
                'arbitrage_value_wkend':sorted_e_wkend_value}
    
    #df_row['e_chrgs_with_PV'] = sum(tariff_results['e_period_charges'])
    #df_row['arbitrage_value_wkday'] = sorted_e_wkday_value
    #df_row['arbitrage_value_wkend'] = sorted_e_wkend_value
    
    return results
    
#%%
def estimate_annual_arbitrage_profit(power, capacity, eta_charge, eta_discharge, sorted_e_wkday_value, sorted_e_wkend_value):

    '''
    This function went with the old 12x24 estimator params    
    
    This function uses the 12x24 marginal energy costs from calc_estimator_params
    to estimate the potential arbitrage value of a battery.
    
    
    To Do
        -Right now it fully cycles every 24 hours, even in situations where
         there is no arbitrage opportunity, resulting in reduced or negative 
         energy values. Definitely set min to zero, and potentially estimate
         better.
        -restrict action if cap > 12*power
    '''
    
    # Started working on better estimation of hours of positive arbitrage value
    # Determine how many hours have positive arbitrage value opportunity
    #wkday_diff = sorted_e_wkday_value[:,np.arange(23,11,-1)]*eta - sorted_e_wkday_value[:,:12]/eta
    #wkend_diff = sorted_e_wkend_value[:,np.arange(23,11,-1)]*eta - sorted_e_wkend_value[:,:12]/eta
    
    #wkday_value_bool = wkday_diff > 0.0
    #wkend_value_bool = wkend_diff > 0.0
    
    #wkday_n_pos = np.sum(wkday_value_bool, 1)
    #wkend_n_pos = np.sum(wkend_value_bool, 1)
    
    #wkday_n_movement = np.max(wkday_n_pos, capacity, 1)

   
    # Determine how many hour 'blocks' the battery will need to consume to charge, 
    #  and what the kWh consumption during those blocks will be
    # reduce capacity by eta, since this is essentially just estimating what can be discharged off a full battery
    
    charge_blocks = np.zeros(int(np.floor(capacity/eta_charge/power)+1))
    charge_blocks[:-1] = np.tile(power,int(np.floor(capacity/eta_charge/power)))
    charge_blocks[-1] = np.mod(capacity/eta_charge,power)
    
    # Determine how many hour 'blocks' the battery will need to cover to discharge,
    #  and what the kWh discharged during those blocks will be
    discharge_blocks = np.zeros(int(np.floor(capacity*eta_discharge/power)+1))
    discharge_blocks[:-1] = np.tile(power,int(np.floor(capacity*eta_discharge/power)))
    discharge_blocks[-1] = np.mod(capacity*eta_discharge,power)
    
    # Determine the max revenue that can be collected by a complete discharge 
    #  into the most expensive hours. Determine the cost of charging from the 
    #  least expensive hours.
    # This will fail if capacity > 12*power
    wkday_daily_revenue = np.sum(sorted_e_wkday_value[:,np.arange(23,23-len(discharge_blocks),-1)]*discharge_blocks[np.arange(len(discharge_blocks)-1,-1,-1)], 1)
    wkend_daily_revenue = np.sum(sorted_e_wkend_value[:,np.arange(23,23-len(discharge_blocks),-1)]*discharge_blocks[np.arange(len(discharge_blocks)-1,-1,-1)], 1)
    wkday_daily_cost = np.sum(sorted_e_wkday_value[:,:len(charge_blocks)]*charge_blocks, 1)
    wkend_daily_cost = np.sum(sorted_e_wkend_value[:,:len(charge_blocks)]*charge_blocks, 1)    
    
    wkday_daily_profit = wkday_daily_revenue - wkday_daily_cost 
    wkend_daily_profit = wkend_daily_revenue - wkend_daily_cost 
    wkday_daily_profit[wkday_daily_profit<0] = 0
    wkend_daily_profit[wkend_daily_profit<0] = 0
    
    # Determine the total annual revenue from discharging, cost from charging, 
    #  and resulting arbitrage profit opportunity   
    #annual_revenue = sum(wkday_daily_revenue*21) + sum(wkend_daily_revenue*8)
    #annual_cost = sum(wkday_daily_cost*21) + sum(wkend_daily_cost*8)
    #annual_arbitrage_profit = annual_revenue - annual_cost
    
    annual_arbitrage_profit = sum(wkday_daily_profit*21) - sum(wkend_daily_profit*8)

    return annual_arbitrage_profit