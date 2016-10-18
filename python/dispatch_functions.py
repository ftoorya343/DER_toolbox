# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 19:45:23 2016

@author: pgagnon
"""

import numpy as np
import tariff_functions as tFuncs
import general_functions as gFuncs

#%%
def calc_min_possible_demands(res, load_profile, d_periods_month, batt, t, month):
    '''
    Function that determines the minimum possible demands that this battery 
    can achieve for a particular month.
    
    Inputs:
    b: battery class object
    t: tariff class object
    
    to-do:
    add a vector of forced discharges, for demand response representation
    
    '''
    # Recast d_periods_month vector into d_periods_index, which is in terms of increasing integers starting at zero
    unique_periods = np.unique(d_periods_month)
    Dn_month = len(unique_periods)
    d_periods_index = np.copy(d_periods_month)
    for n in range(len(unique_periods)): d_periods_index[d_periods_month==unique_periods[n]] = n
     
     
    # Calculate the original and minimum possible demands in each period
    original_demands = np.zeros(Dn_month)
    min_possible_demands = np.zeros(Dn_month)
    for period in range(Dn_month):
        original_demands[period] = np.max(load_profile[d_periods_index==period])
        min_possible_demands = original_demands - batt.power
            
    # d_ranges is the range of demands in each period that will be investigated
    d_ranges = np.zeros((res,Dn_month), float)
    for n in range(Dn_month):
        d_ranges[:,n] = np.linspace(min_possible_demands[n], original_demands[n], res)
        
    # First evaluate a set that cuts diagonally across the search space
    # I haven't checked to make sure this is working properly yet
    # At first glance, the diagonal seems to slow quite a bit if period=1, so maybe identify and skip if p=1?
    for n in range(len(d_ranges[:,0])):
        success = dispatch_pass_fail(load_profile, d_periods_index, d_ranges[n,:], batt)
        if success == True: 
            i_of_first_success = n
            break
        
    # Assemble a list of all combinations of demand levels within the ranges of 
    # interest, calculate their demand charges, and sort by increasing cost
    d_combinations = np.zeros(((res-i_of_first_success)**len(unique_periods),Dn_month+1), float)
    d_combinations[:,:Dn_month] = gFuncs.cartesian([np.asarray(d_ranges[i_of_first_success:,x]) for x in range(Dn_month)])
    TOU_demand_charge = np.sum(tFuncs.tiered_calc_vec(d_combinations[:,:Dn_month], t.d_tou_levels[:,unique_periods], t.d_tou_prices[:,unique_periods]),1) #check that periods line up with rate
    monthly_demand_charge = tFuncs.tiered_calc_vec(np.max(d_combinations[:,:Dn_month],1), t.d_flat_levels[:,month], t.d_flat_prices[:,month])
    d_combinations[:,-1] = TOU_demand_charge + monthly_demand_charge   
    
    # this old approach left a zero-cost option essentially unsorted
    #d_combinations = d_combinations[d_combinations[:,-1].argsort(order = ['cost','p1'])]
    
    # These next several steps are sorting first by ascending cost, and then
    # by descending value of the lowest cost period (in an effort to catch a zero-cost period)
    # Without this, it would solve for the highest-demand zero-cost period value that
    # could still meet the other period's requirements, which would then result in unnecessary 
    # dispatching to peak shaving during zero-cost periods. 
    # invert sign so that the sorting will work...
    d_combinations[:,:-1] = -d_combinations[:,:-1]
    
    d_tou_costs = np.sum(t.d_tou_prices[:,unique_periods], 0)
    i_min = np.argmin(d_tou_costs)
    d_combinations = d_combinations[np.lexsort((d_combinations[:,i_min], d_combinations[:,-1]))]

    # invert back...
    d_combinations[:,:-1] = -d_combinations[:,:-1]
    
    cheapest_d_states = np.zeros(t.d_tou_n+1)
    for n in range(len(d_combinations[:,0])):
        success = dispatch_pass_fail(load_profile, d_periods_index, d_combinations[n,:-1], batt)
        if success == True:
            #print "Cheapest possible demand states:", np.round(d_combinations[n,:],1)
            cheapest_d_states[unique_periods] = d_combinations[n,:Dn_month]
            cheapest_d_states[-1] = d_combinations[n,-1]
            break
    
    d_max_vector = np.zeros(len(d_periods_month))
    # not correct
    for p in unique_periods: d_max_vector[d_periods_month==p] = cheapest_d_states[p]
    #for n in range(len(unique_periods)): d_max_vector[Dper==unique_periods[n]] = cheapest_d_states[n]
    #cheapest_d_states_all[unique_periods] = cheapest_d_states[]
    
    # Report original demand levels and their total (TOU+seasonal) cost, for reference bill calculation
    
    
    return cheapest_d_states, d_max_vector
    
#%%
def dispatch_pass_fail(load_profile, demand_periods_vector, targets, batt):
    '''
    Function to determine if the battery with the specified power, capacity, and efficiency
    values can achieve the demand levels specified.
    
    INPUTS:
    load_profile: Original load profile, that the battery will be manipulating
    demand_periods_vector = Vector of integer demand periods, equal to the length
                        of the time period under evaluation (typically the hours in a month)
    targets = 
    
    '''
    demand_vector = [targets[d] for d in demand_periods_vector] # Map the target demands to the hourly vector
    poss_batt_level_change = demand_vector - load_profile
    poss_batt_level_change = [batt.power if s>batt.power else s for s in poss_batt_level_change] 
    poss_batt_level_change = [-batt.power if s<-batt.power else s for s in poss_batt_level_change] 
    poss_batt_level_change = [s/batt.eta if s<0 else s*batt.eta for s in poss_batt_level_change]
    batt_e_level = batt.cap
    
    success = True
    for n in range(len(demand_periods_vector)):
        batt_e_level += poss_batt_level_change[n]
        if batt_e_level < 0: success=False; break
        elif batt_e_level > batt.cap: batt_e_level = batt.cap
    
    return success

#%% Energy Arbitrage Value Estimator
def calc_estimator_params(load_and_pv_profile, tariff, NEM_tariff, eta):
    '''
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
    e_period_sums = tariff_results['e_period_sums']
    e_charges = tariff_results['e_period_charges']
    
    # Add 1 kW to every hour of the load profile and recalculate the new energy charges
    load_and_pv_profile_plus_1kw = load_and_pv_profile + 1
    annual_bill, tariff_results2 = tFuncs.bill_calculator(load_and_pv_profile_plus_1kw, tariff, NEM_tariff)
    e_period_sums2 = tariff_results2['e_period_sums']
    e_charges2 = tariff_results2['e_period_charges']
    
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

    results = {'e_chrgs_with_PV':sum(tariff_results['e_period_charges']),
                'arbitrage_value_wkday':sorted_e_wkday_value,
                'arbitrage_value_wkend':sorted_e_wkend_value}
    
    #df_row['e_chrgs_with_PV'] = sum(tariff_results['e_period_charges'])
    #df_row['arbitrage_value_wkday'] = sorted_e_wkday_value
    #df_row['arbitrage_value_wkend'] = sorted_e_wkend_value
    
    return results
    
#%%
def estimate_annual_arbitrage_profit(power, capacity, eta, sorted_e_wkday_value, sorted_e_wkend_value):

    '''
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
    
    charge_blocks = np.zeros(int(np.floor(capacity/eta/power)+1))
    charge_blocks[:-1] = np.tile(power,int(np.floor(capacity/eta/power)))
    charge_blocks[-1] = np.mod(capacity/eta,power)
    
    # Determine how many hour 'blocks' the battery will need to cover to discharge,
    #  and what the kWh discharged during those blocks will be
    discharge_blocks = np.zeros(int(np.floor(capacity*eta/power)+1))
    discharge_blocks[:-1] = np.tile(power,int(np.floor(capacity*eta/power)))
    discharge_blocks[-1] = np.mod(capacity*eta,power)
    
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