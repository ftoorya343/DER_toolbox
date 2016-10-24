# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 19:45:23 2016

@author: pgagnon
"""

import numpy as np
import tariff_functions as tFuncs
import general_functions as gFuncs


class Battery:
    
    def __init__(self, cap=0.0, power=0.0, SOC_min=0.2, eta_charge=0.91, eta_discharge=0.91):
        self.SOC_min = SOC_min    
        self.power = power
        self.cap = cap
        self.cap_effective = cap*(1-SOC_min)
        self.eta_charge = eta_charge
        self.eta_discharge = eta_charge
    
    def set_cap_and_power(self, cap, power):
        self.power = power
        self.cap = cap
        self.cap_effective = cap*(1-self.SOC_min)

#%%
def determine_optimal_dispatch(load_profile, batt, t, export_tariff, d_inc_n=50, DP_inc=50, estimator_params=None, estimated=False):
    '''
    Function that determines the optimal dispatch of the battery, and in the
    process determines the resulting first year bill with the system.
    
    INPUTS:
    estimate_toggle: Boolean. False means run DP to get accurate energy savings
                     and dispatch trajectory. True means estimate the energy
                     savings, and don't get the dispatch trajectory.
                     
    load_profile: Original load profile prior to modification by the battery
                  (It includes PV's contribution, if there is any)
    
    t: tariff class object
    b: battery class object
    
    NOTES:
    -in the battery level matrices, 0 index corresponds to an empty battery, and 
     the highest index corresponds to a full battery
    
    To Do:
    -Make it evaluate the bill for the net profile when batt.cap == 0
    -Having cost-to-go equal cost of filling the battery at the end may not be
     working.
    -have warnings for classes of errors. Same for bill calculator, such as when
     net load in a given period is negative
    -either have warnings, or outright nans, when an illegal move is chosen
    -I see some demand max violations caused by minor dips below the demand_max
     line, when they occur in an otherwise demanding profile. This can happen
     when there is a large PV system that causes a mid-day dip. Solution: manually
     construct an offset vector by reverse cumSum, setting at zero for empty batt
    -DP trajectory seems mostly correct, except the preference for low-power
     trajectories doesn't seem to be enforced
    -If there are no demand charges, don't calc & don't have a limit on 
     demand_max_profile for the following dispatch.
    
    
    '''
    if batt.cap == 0.0:
        opt_load_traj = load_profile
        bill_under_dispatch, _ = tFuncs.bill_calculator(opt_load_traj, t, export_tariff)
        demand_max_exceeded = False

    else:
        # =================================================================== #
        # Determine cheapest possible demand states for the entire year
        # =================================================================== #
        #d_inc_n = 100 # Number of demand levels between original and lowest possible that will be explored
        month_hours = np.array([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]);
        cheapest_possible_demands = np.zeros((12,t.d_tou_n+1), float)
        demand_max_profile = np.zeros(len(load_profile), float)
        
        # Determine the cheapest possible set of demands for each month, and create an annual profile of those demands
        for month in range(12):
            # Extract the load profile for only the month under consideration
            load_profile_month = load_profile[month_hours[month]:month_hours[month+1]]
            d_tou_month_periods = t.d_tou_8760[month_hours[month]:month_hours[month+1]]
            
            # columns [:-1] of cheapest_possible_demands are the achievable demand levels, column [-1] is the cost
            # d_max_vector is an hourly vector of the demand level of that period (to become a max constraint in the DP), which is cast into an 8760 for the year.
            cheapest_possible_demands[month,:], d_max_vector = calc_min_possible_demands(d_inc_n, load_profile_month, d_tou_month_periods, batt, t, month)
            demand_max_profile[month_hours[month]:month_hours[month+1]] = d_max_vector
            
        
        # =================================================================== #
        # Complete (not estimated) dispatch of battery with dynamic programming    
        # =================================================================== #
        if estimated == False:    
            DP_res = batt.cap / (DP_inc-1)
            illegal = 99999999
            
            batt_influence_to_achieve_demand_max = demand_max_profile - load_profile
            batt_influence_to_achieve_demand_max = np.clip(batt_influence_to_achieve_demand_max, -batt.power, batt.power) #The clip might not be necessary, since I mod anyway. 
            
            batt_actions_to_achieve_demand_max = np.array([s*batt.eta_charge if s >= 0 else s/batt.eta_discharge for s in batt_influence_to_achieve_demand_max], float)
            
            # Build offset
            batt_e_levels = np.zeros(len(batt_actions_to_achieve_demand_max), float)
            batt_e_levels[-1] = 0 #or maybe zero?
            for hour in np.arange(len(batt_actions_to_achieve_demand_max)-2,-1,-1):
                batt_e_levels[hour] = batt_e_levels[hour+1] - batt_actions_to_achieve_demand_max[hour+1]
                if batt_e_levels[hour] < 0: 
                    batt_e_levels[hour] = 0 # This might not be necessary
                elif batt_e_levels[hour] > batt.cap: 
                    batt_e_levels[hour] = batt.cap
            
            batt_actions_to_achieve_demand_max = np.zeros(len(batt_actions_to_achieve_demand_max), float)
            for hour in np.arange(len(batt_actions_to_achieve_demand_max)-2,-1,-1):
                batt_actions_to_achieve_demand_max[hour+1] = batt_e_levels[hour+1] - batt_e_levels[hour]
            
            batt_act_cumsum_mod_rev = np.mod(np.cumsum(batt_actions_to_achieve_demand_max[np.arange(8759,-1,-1)])[np.arange(8759,-1,-1)], DP_res)
                
            # Casting a wide net, and doing a pass/fail test later on with cost-to-go. May later evaluate the limits up front.
            batt_charge_limit = int(batt.power*batt.eta_charge/DP_res) + 1
            batt_discharge_limit = int(batt.power/batt.eta_discharge/DP_res) + 1
            batt_charge_limits_len = batt_charge_limit + batt_discharge_limit + 1
            # the fact the battery row levels aren't anchored anymore hasn't been thought through. How will I make sure my net is aligned?
            
            batt_levels_n = DP_inc # probably the same as expected_values_n
            batt_levels_temp = np.zeros([batt_levels_n,8760])
            batt_levels_temp[:,:] = np.linspace(0,batt.cap,batt_levels_n, float).reshape(batt_levels_n,1)
            
            batt_levels_shift = batt_levels_temp.copy()
            batt_levels_shift[:,:-1] = batt_levels_temp[:,:-1] + (DP_res - batt_act_cumsum_mod_rev[1:].reshape(1,8759)) #haven't checked batt_act_cumsum_mod
            
            batt_levels = np.zeros([batt_levels_n+1,8760], float)
            batt_levels[1:,:] = batt_levels_shift
            batt_levels[0,:] = 0.0
            batt_levels[-1,:] = batt.cap
            #batt_levels = np.clip(batt_levels, 0, batt.cap)
            
            
            batt_levels_buffered = np.zeros([np.shape(batt_levels)[0]+batt_charge_limit+batt_discharge_limit, np.shape(batt_levels)[1]], float)
            batt_levels_buffered[:batt_discharge_limit,:] = illegal
            batt_levels_buffered[-batt_charge_limit:,:] = illegal
            batt_levels_buffered[batt_discharge_limit:-batt_charge_limit,:] = batt_levels
            
            base_change_in_batt_level_vector = np.zeros(batt_discharge_limit+batt_charge_limit+1, float)
            base_change_in_batt_level_vector[:batt_discharge_limit+1] = np.linspace(-batt_discharge_limit*DP_res,0,batt_discharge_limit+1, float)
            base_change_in_batt_level_vector[batt_discharge_limit:] = np.linspace(0,batt_charge_limit*DP_res,batt_charge_limit+1, float)
            
            # Each row corresponds to a battery level, each column is the change in batt level associated with that movement.
            # The first row corresponds to an empty batt. So it shouldn't be able to discharge. 
            # So maybe filter by resulting_batt_level, and exclude ones that are negative or exceed cap?
            base_change_in_batt_level_matrix = np.zeros((batt_levels_n+1, len(base_change_in_batt_level_vector)), float)
            change_in_batt_level_matrix = np.zeros((batt_levels_n+1, len(base_change_in_batt_level_vector)), float)
            base_change_in_batt_level_matrix[:,:] = base_change_in_batt_level_vector
            
            # Build an adjust to prefer slow charge, all else being equal
            adjuster = np.zeros(batt_charge_limits_len, float)
            base_adjustment = 0.0000001            
            adjuster[np.arange(batt_discharge_limit,-1,-1)] = base_adjustment * np.array(range(batt_discharge_limit+1))*np.array(range(batt_discharge_limit+1)) / (batt_discharge_limit*batt_discharge_limit)
            adjuster[batt_discharge_limit:] = base_adjustment * np.array(range(batt_charge_limit+1))*np.array(range(batt_charge_limit+1)) / (batt_charge_limit*batt_charge_limit)

            
            # Initialize some objects for later use in the DP
            expected_value_n = DP_inc+1
            expected_values = np.zeros((expected_value_n, np.size(load_profile)), float)
            DP_choices = np.zeros((DP_inc+1, np.size(load_profile)), int)
            influence_on_load = np.zeros(np.shape(base_change_in_batt_level_matrix), float)
            selected_net_loads = np.zeros((DP_inc+1, np.size(load_profile)), float)
            
            # Expected value of final states is the energy required to fill the battery up
            # at the most expensive electricity rate. This encourages ending with a full
            # battery, but solves a problem of demand levels being determined by a late-hour
            # peak that the battery cannot recharge from before the month ends
            # This would be too strict under a CPP rate.
            # I should change this to evaluating the required charge based on the batt_level matrix, to keep self-consistent
            expected_values[:,-1] = np.linspace(batt.cap,0,DP_inc+1)/batt.eta_charge*np.max(t.e_prices_no_tier) #this should be checked, after removal of buffer rows
            
            # Each row is the set of options for a single battery state
            # Each column is an index corresponding to the possible points within the expected_value matrix that that state can reach
            option_indicies = np.zeros((DP_inc+1, batt_charge_limits_len), int)
            option_indicies[:,:] = range(batt_charge_limits_len)
            for n in range(DP_inc+1):
                option_indicies[n,:] += n - batt_discharge_limit
            option_indicies = (option_indicies>0) * option_indicies # have not checked this default of pointing towards zero
            option_indicies = (option_indicies<DP_inc+1) * option_indicies
            
            net_loads = np.zeros((DP_inc+1, batt_charge_limits_len), float)
            costs_to_go = np.zeros((DP_inc+1, batt_charge_limits_len), float)
            
            
            ###################################################################
            ############### Dynamic Programming Energy Trajectory #############
            
            for hour in np.arange(np.size(load_profile)-2, -1, -1):
                # Rows correspond to each possible battery state
                # Columns are options for where this particular battery state could go to
                # Index is hour+1 because the DP decisions are on a given hour, looking ahead to the next hour. 
            
                # this is beginning of a quicker approach to just adjust the base matrix
                #change_in_batt_level_matrix = base_change_in_batt_level_matrix + batt_act_cumsum_mod[hour+1] - batt_act_cumsum_mod[hour]
                
                # this is just an inefficient but obvious way to assembled this matrix. It should be possible in a few quicker operations.
                for row in range(batt_levels_n+1):
                    change_in_batt_level_matrix[row,:] = (-batt_levels[row,hour] + batt_levels_buffered[row:row+batt_charge_limits_len,hour+1])
                    
                #Because of the 'illegal' values, neg_batt_bool shouldn't be necessary
                resulting_batt_level = change_in_batt_level_matrix + batt_levels[:,hour].reshape(DP_inc+1,1)
                neg_batt_bool = resulting_batt_level<0
                overfilled_batt_bool = resulting_batt_level>batt.cap #this seems to be misbehaving due to float imprecision
                                
                charging_bool = change_in_batt_level_matrix>0
                discharging_bool = change_in_batt_level_matrix<0
                
                influence_on_load = np.zeros(np.shape(change_in_batt_level_matrix), float)
                influence_on_load += (change_in_batt_level_matrix*batt.eta_discharge) * discharging_bool
                influence_on_load += (change_in_batt_level_matrix/batt.eta_charge) * charging_bool
                influence_on_load -= 0.000000001 # because of rounding error? Problems definitely occur (sometimes) without this adjustment. The adjustment magnitude has not been tuned since moving away from ints.
                
                net_loads = load_profile[hour+1] + influence_on_load
            
                # I may also need to filter for moves the battery can't actually make
                
                # Determine the incremental cost-to-go for each option
                costs_to_go[:,:] = 0 # reset costs to go
                importing_bool = net_loads>=0 # If consuming, standard price
                costs_to_go += net_loads*t.e_prices_no_tier[t.e_tou_8760[hour+1]]*importing_bool
                exporting_bool = net_loads<0 # If exporting, NEM price
                costs_to_go += net_loads*export_tariff.prices[export_tariff.periods_8760[hour+1]]*exporting_bool     
                
                # Make the incremental cost of impossible/illegal movements very high
                costs_to_go += neg_batt_bool * illegal
                costs_to_go += overfilled_batt_bool * illegal
                demand_limit_exceeded_bool = net_loads>demand_max_profile[hour+1]
                costs_to_go += demand_limit_exceeded_bool * illegal
                
                # add very small cost as a function of battery motion, to discourage unnecessary motion
                costs_to_go += adjuster
                    
                total_option_costs = costs_to_go + expected_values[option_indicies, hour+1]
                
                # something is wrong - in the final step I see an optimal choice of having
                # a partially discharged battery, instead of having a full battery. 
                # It should be nearly identical, but nonetheless not preferrable to have an empty
                # I think it may have something to do with the mapping    
                
                expected_values[:, hour] = np.min(total_option_costs,1)     
                     
                #Each row corresponds to a row of the battery in DP_states. So the 0th row are the options of the empty battery state.
                #The indicies of the results correspond to the battery's movement. So the (approximate) middle option is the do-nothing option   
                #Subtract the negative half of the charge vector, to get the movement relative to the row under consideration        
                DP_choices[:,hour] = np.argmin(total_option_costs,1) - batt_discharge_limit # adjust by discharge?
                selected_net_loads[:,hour] = net_loads[range(DP_inc+1),np.argmin(total_option_costs,1)]
                
            # Determine what the indexes of the optimal trajectory were.
            # Start at the 0th hour, imposing a full battery.
            # traj_i is the indexes of the battery's trajectory.
            traj_i = np.zeros(len(load_profile), int)
            traj_i[0] = DP_inc-1
            for n in range(len(load_profile)-1):
                traj_i[n+1] = traj_i[n] + DP_choices[int(traj_i[n]), n]
            
            opt_load_traj = np.zeros(len(load_profile), float)
            for n in range(len(load_profile)-1):
                opt_load_traj[n+1] = selected_net_loads[traj_i[n], n] 
                
            opt_batt_traj = np.zeros(len(load_profile))
            opt_batt_traj_f = np.zeros(len(load_profile))
            for n in range(len(load_profile)-1):
                opt_batt_traj[n] = batt_levels[traj_i[n], n]
                
                
            batt_movement = np.zeros(len(load_profile))
            for n in np.arange(1,len(load_profile)-1,1):
                batt_movement[n] = opt_batt_traj[n] - opt_batt_traj[n-1]
                
            batt_influence_on_load = np.array([s/batt.eta_charge if s >= 0 else s*batt.eta_discharge for s in batt_movement], float)
            
            opt_net_profile = load_profile + batt_influence_on_load
            
            # This isn't strictly necessary
            bill_under_dispatch, _ = tFuncs.bill_calculator(opt_load_traj, t, export_tariff)
            demand_max_exceeded = np.any(opt_load_traj[1:] > demand_max_profile[1:])
            
        
        elif estimated == True:
            batt_arbitrage_value = estimate_annual_arbitrage_profit(batt.power, batt.cap_effective, batt.eta_charge, batt.eta_discharge, estimator_params['cost_sum'], estimator_params['revenue_sum'])                
            bill_under_dispatch = sum(cheapest_possible_demands[:,-1]) + 12*t.fixed_charge + estimator_params['e_chrgs_with_PV'] - batt_arbitrage_value
            opt_load_traj = None
            #energy_charges = estimator_params['e_chrgs_with_PV'] - batt_arbitrage_value
            demand_max_exceeded = False

    ######################### Package Results #################################
    
    results = {'opt_load_profile':opt_load_traj,
               'bill_under_dispatch':bill_under_dispatch,
               'demand_max_exceeded':demand_max_exceeded}
               
    return results

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
    poss_batt_level_change = [s/batt.eta_discharge if s<0 else s*batt.eta_charge for s in poss_batt_level_change]
    batt_e_level = batt.cap
    
    success = True
    for n in range(len(demand_periods_vector)):
        batt_e_level += poss_batt_level_change[n]
        if batt_e_level < 0: success=False; break
        elif batt_e_level > batt.cap: batt_e_level = batt.cap
    
    return success

    
#%% Energy Arbitrage Value Estimator
def calc_estimator_params(load_and_pv_profile, tariff, export_tariff, eta_charge, eta_discharge):
    '''
    This function creates four 12-length vectors, weekend/weekday and 
    cost/revenue. They are a summation of each day's 12 hours of lowest/highest
    cost electricity.
    
    Assumptions:
        -TOU windows are aligned with when the battery would be dispatching for
         demand peak shaving.
        -The battery will be able to dispatch fully and recharge fully every 24
         hour cycle.
    
    To Do:
        -Bring back consideration of tiers.
        -Consider coming up with a better method that captures exportation, CPP, etc
         Maybe? Or just confirm a simple estimation works with our dGen set, 
         and use the accurate dispatch for any other analysis.
    
    '''
    
    # Calculate baseline energy costs with the given load+pv profile
    _, tariff_results = tFuncs.bill_calculator(load_and_pv_profile, tariff, export_tariff)
    e_chrgs_with_PV = tariff_results['e_charges']
    
    # Estimate the marginal retail energy costs of each hour
    e_value_8760 = np.average(tariff.e_prices, 0)[tariff.e_tou_8760]
    e_value_8760[load_and_pv_profile<=0] = np.average(export_tariff.prices, 0)[export_tariff.periods_8760][load_and_pv_profile<=0]
    
    # Reshape into 365 24-hour day vectors and then sort by increasing cost
    e_value_365_24 = e_value_8760.reshape((365,24), order='C')
    e_value_365_24_sorted = np.sort(e_value_365_24)
    
    # Split the lower half into costs-to-charge and upper half into revenue-from-discharge
    e_cost = e_value_365_24_sorted[:,:12]
    e_revenue = e_value_365_24_sorted[:,np.arange(23,11,-1)]
    
    # Estimate which hours there is actually an arbitrage profit, where revenue
    #  exceeds costs for a pair of hours in a day. Not strictly correct, because
    #  efficiencies means that hours are not directly compared.
    arbitrage_opportunity = e_revenue*eta_discharge > e_cost*eta_charge
    
    # Where there is no opportunity, replace both cost and revenue values with
    #  0, to reflect no battery action in those hours.
    e_cost[arbitrage_opportunity==False] = 0.0 
    e_revenue[arbitrage_opportunity==False] = 0.0 
    
    cost_sum = np.sum(e_cost, 0)
    revenue_sum = np.sum(e_revenue, 0)

    results = {'e_chrgs_with_PV':e_chrgs_with_PV,
                'cost_sum':cost_sum,
                'revenue_sum':revenue_sum}
    
    return results
    
#%%
def estimate_annual_arbitrage_profit(power, capacity, eta_charge, eta_discharge, cost_sum, revenue_sum):

    '''
    This function uses the 12x24 marginal energy costs from calc_estimator_params
    to estimate the potential arbitrage value of a battery.
    
    Inputs:
    -cost_sum: 12-length sorted vector of summed energy costs for charging in
     the cheapest 12 hours of each day
    -revenue_sum: 12-length sorted vector of summed energy revenue for
     discharging in the most expensive 12 hours of each day
    
    
    To Do
        -restrict action if cap > 12*power
    '''
    
    charge_blocks = np.zeros(12)
    charge_blocks[:int(np.floor(capacity/eta_charge/power))] = power
    charge_blocks[int(np.floor(capacity/eta_charge/power))] = np.mod(capacity/eta_charge,power)  
    
    # Determine how many hour 'blocks' the battery will need to cover to discharge,
    #  and what the kWh discharged during those blocks will be
    discharge_blocks = np.zeros(12)
    discharge_blocks[:int(np.floor(capacity*eta_discharge/power)+1)] = power
    discharge_blocks[int(np.floor(capacity*eta_discharge/power)+1)] = np.mod(capacity*eta_discharge,power)
        
    revenue = np.sum(revenue_sum * eta_discharge * discharge_blocks)
    cost = np.sum(cost_sum * eta_charge * charge_blocks)
    
    annual_arbitrage_profit = revenue - cost

    return annual_arbitrage_profit