# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 00:03:41 2016

@author: pgagnon
"""








this script has been retired, and moved into dispatch functions




















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


'''

import numpy as np
import os
import tariff_functions as tFuncs
import dispatch_functions as dFuncs
import general_functions as gFuncs
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt, mpld3


t = tFuncs.Tariff(json_file_name='pge_e20.json')

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
    
profile = np.genfromtxt('input_profile_reopt_compare_sf.csv', delimiter=",", skip_header=1)
original_load_profile = profile[:,0]
pv_cf_profile = profile[:,1]

pv_size = 2088.63
load_profile = original_load_profile - pv_size*pv_cf_profile
pv_profile = pv_size*pv_cf_profile
aep = np.sum(pv_profile)
aec = np.sum(original_load_profile)
energy_penetration = aep / aec
print "annual energy penetration:", energy_penetration



# ======================================================================= #
# Determine cheapest possible demand states for the entire year
# ======================================================================= #
d_inc_n = 100 # Number of demand levels between original and lowest possible that will be explored
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
    cheapest_possible_demands[month,:], d_max_vector = dFuncs.calc_min_possible_demands(d_inc_n, load_profile_month, d_tou_month_periods, batt, t, month)
    demand_max_profile[month_hours[month]:month_hours[month+1]] = d_max_vector
    

# =================================================================== #
# Dynamic programming dispatch for energy trajectory      
# =================================================================== #
DP_inc = 100
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
    

#%%
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

###############################################################################
# Bring the adjuster back later
###############################################################################
## Slightly adjust the cost/benefits of movement with higher penalities
## for higher power action, such that the DP will prefer constant low
## power charge/discharge over high power discharge
#adjuster = np.zeros(len(base_influence_on_net_load))
#for n in range(batt_discharge_limit): adjuster[n] = 0.00001 * (1.0-(n)/float(batt_discharge_limit))**2.0
#for n in range(batt_charge_limit): adjuster[n+1+batt_discharge_limit] = 0.00001 * ((n+1)/float(batt_charge_limit))**2.0

adjuster = np.zeros(batt_charge_limits_len, float)
base_adjustment = 1.0
for n in range(batt_discharge_limit): adjuster[n] = base_adjustment * (1.0-(n)/float(batt_discharge_limit))**2.0
for n in range(batt_charge_limit): adjuster[n+1+batt_discharge_limit] = base_adjustment * ((n+1)/float(batt_charge_limit))**2.0

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
costs_to_go = np.zeros((DP_inc+1, batt_charge_limits_len), float) # should clean up the DP_inc+1 at some point...


#%%
# Dynamic Programming Energy Trajectory
for hour in np.arange(np.size(load_profile)-2, -1, -1):
    # Rows correspond to each possible battery state
    # Columns are options for where this particular battery state could go to
    # Index is hour+1 because the DP decisions are on a given hour, looking ahead to the next hour. 

    # this is beginning of a quicker approach to just adjust the base matrix
    #change_in_batt_level_matrix = base_change_in_batt_level_matrix + batt_act_cumsum_mod[hour+1] - batt_act_cumsum_mod[hour]
    
    # this is just an inefficient but obvious way to assembled this matrix. It should be possible in a few quicker operations.
    for row in range(batt_levels_n+1):
        change_in_batt_level_matrix[row,:] = (-batt_levels[row,hour] + batt_levels_buffered[row:row+batt_charge_limits_len,hour+1])

    resulting_batt_level = change_in_batt_level_matrix + batt_levels[:,hour].reshape(DP_inc+1,1)
    neg_batt_bool = resulting_batt_level<0
    overfilled_batt_bool = resulting_batt_level>batt.cap #this seems to be misbehaving due to float imprecision
    
    adjuster = (change_in_batt_level_matrix / np.max(change_in_batt_level_matrix[33,:]))**2.0 #* 0.0001    
    
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
    
# Determine what the optimal trajectory was
# Start at the 0th hour, imposing a full battery    
# traj_i is the battery's trajectory.
traj_i = np.zeros(len(load_profile), int)
traj_i[0] = DP_inc-1
for n in range(len(load_profile)-1):
    traj_i[n+1] = traj_i[n] + DP_choices[int(traj_i[n]), n]
    
# There is a problem rebuilding the net profile from the battery trajectory
# I should check that integrating the various trajectories agree with each other
    
# I also see occasional points where it is discharging more than it strictly needs to, which it shouldn't do due to
# adjuster penalty for action
   

opt_load_traj = np.zeros(len(load_profile), float)
for n in range(len(load_profile)-1):
    opt_load_traj[n+1] = selected_net_loads[traj_i[n], n] 
    
opt_batt_traj = np.zeros(len(load_profile))
opt_batt_traj_f = np.zeros(len(load_profile))
for n in range(len(load_profile)-1):
#    opt_batt_traj_f[n] = batt_levels[traj_i[n], n]
#    opt_batt_traj[n] = opt_batt_traj_f[n] * batt.cap
    opt_batt_traj[n] = batt_levels[traj_i[n], n]
    
    
batt_movement = np.zeros(len(load_profile))
for n in np.arange(1,len(load_profile)-1,1):
    batt_movement[n] = opt_batt_traj[n] - opt_batt_traj[n-1]
    
batt_influence_on_load = np.array([s/batt.eta_charge if s >= 0 else s*batt.eta_discharge for s in batt_movement], float)

opt_net_profile = load_profile + batt_influence_on_load

print "Demand Max Exceeded:", np.any(opt_load_traj[1:] > demand_max_profile[1:])
#%%
e_price_vec = np.zeros(8760)
for n in range(8760):
    e_price_vec[n] = t.e_prices[0][t.e_tou_8760[n]]

time = range(8760)
plt.figure(figsize=(20,8))
#plt.figure(figsize=(8,3))
plt.plot(time, load_profile, 'black', linewidth=2)
plt.plot(time, demand_max_profile, 'red', linewidth=2)
plt.plot(time, opt_load_traj, 'blue', linewidth=2)
#plt.fill_between(time, np.zeros(8760), e_price_vec*max(load_profile)/max(e_price_vec), facecolor='yellow', alpha=0.5) 
plt.legend(['Load', 'demand max', 'retrieved opt', 'e tou'])
plt.grid(True)
mpld3.show()






## Determine what the energy consumption for each period was
## Each row is a month, each column is a period
#energy_consumpt = np.zeros([12,t.e_n])
#for month in range(12):
#    net_profile_month = net_profile[month_hours[month]:month_hours[month+1]]
#    energy_periods_month = t.e_tou_8760[month_hours[month]:month_hours[month+1]]
#    for period in range(t.e_n):
#        net_periods = net_profile_month[energy_periods_month==period]
#        #imported = net_periods[net_periods>0]
#        #just assuming full retail NEM for now...
#        energy_consumpt[month,period] = sum(net_periods)
#
#energy_charges = tiered_calc_vec(energy_consumpt, t.e_levels[:,:t.e_n], t.e_prices[:,:t.e_n])
#
#final_bill = sum(cheapest_possible_demands[:,-1]) + np.sum(energy_charges) + 12*t.fixed_charge


