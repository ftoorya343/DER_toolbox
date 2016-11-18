# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 16:13:43 2016

@author: pgagnon
"""

import numpy as np
import pandas as pd
import demand_functions as demFun
import rate_import_functions as rate_import
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


#%%
def calc_min_possible_demands(res, load_profile_year, batt, tariff):
    '''
    Function that determines the minimum possible demands that this battery 
    can achieve for a particular month.
    
    Inputs:
    b: battery class object
    t: tariff class object
    
    to-do:
    add a vector of forced discharges, for demand response representation
    
    '''

    month_hours = np.array([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]) * len(load_profile_year)/8760;
    cheapest_possible_demands = np.zeros((12,tariff.d_n), float)
    charges_for_cheapest_demands = np.zeros(12, float)
    demand_max_profile = np.zeros(len(load_profile_year), float)
    
    # Determine the cheapest possible set of demands for each month, and create an annual profile of those demands
    for month in range(12):
        # Extract the load profile for only the month under consideration
        load_profile_month = load_profile_year[month_hours[month]:month_hours[month+1]]
        d_periods_month = tariff.tou_schedule[month_hours[month]:month_hours[month+1]]    
    
        # Recast d_periods_month vector into d_periods_index, which is in terms of increasing integers starting at zero
        unique_periods = np.unique(d_periods_month)
        Dn_month = len(unique_periods)
        d_periods_index = np.copy(d_periods_month)
        for n in range(len(unique_periods)): d_periods_index[d_periods_month==unique_periods[n]] = n
         
        # Calculate the original and minimum possible demands in each period
        original_demands = np.zeros(Dn_month)
        min_possible_demands = np.zeros(Dn_month)
        for period in range(Dn_month):
            original_demands[period] = np.max(load_profile_month[d_periods_index==period])
            min_possible_demands = original_demands - batt.power
                
        # d_ranges is the range of demands in each period that will be investigated
        d_ranges = np.zeros((res,Dn_month), float)
        for n in range(Dn_month):
            d_ranges[:,n] = np.linspace(min_possible_demands[n], original_demands[n], res)
            
        # First evaluate a set that cuts diagonally across the search space
        # I haven't checked to make sure this is working properly yet
        # At first glance, the diagonal seems to slow quite a bit if period=1, so maybe identify and skip if p=1?
        for n in range(len(d_ranges[:,0])):
            success = dispatch_pass_fail(load_profile_month, d_periods_index, d_ranges[n,:], batt)
            if success == True: 
                i_of_first_success = n
                break
            
        # Assemble a list of all combinations of demand levels within the ranges of 
        # interest, calculate their demand charges, and sort by increasing cost
        d_combinations = np.zeros(((res-i_of_first_success)**len(unique_periods),Dn_month+1), float)
        d_combinations[:,:Dn_month] = cartesian([np.asarray(d_ranges[i_of_first_success:,x]) for x in range(Dn_month)])
        TOU_demand_charge = np.sum(tiered_calc_vec(d_combinations[:,:Dn_month], tariff.tou_levels[:,unique_periods], tariff.tou_prices[:,unique_periods]),1) #check that periods line up with rate
        monthly_demand_charge = tiered_calc_vec(np.max(d_combinations[:,:Dn_month],1), tariff.seasonal_levels[:,month], tariff.seasonal_prices[:,month])
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
        
        d_tou_costs = np.sum(tariff.tou_prices[:,unique_periods], 0)
        i_min = np.argmin(d_tou_costs)
        d_combinations = d_combinations[np.lexsort((d_combinations[:,i_min], d_combinations[:,-1]))]
    
        # invert back...
        d_combinations[:,:-1] = -d_combinations[:,:-1]
        
        # Find the cheapest possible combination of demand levels. Check the
        # cheapest, if it fails, check the next cheapest, etc...
        cheapest_d_states = np.zeros(tariff.d_n+1)
        for n in range(len(d_combinations[:,0])):
            success = dispatch_pass_fail(load_profile_month, d_periods_index, d_combinations[n,:-1], batt)
            if success == True:
                cheapest_d_states[unique_periods] = d_combinations[n,:Dn_month]
                cheapest_d_states[-1] = d_combinations[n,-1]
                break
        
        d_max_vector = np.zeros(len(d_periods_month))
        # not correct
        for p in unique_periods: d_max_vector[d_periods_month==p] = cheapest_d_states[p]
        #for n in range(len(unique_periods)): d_max_vector[Dper==unique_periods[n]] = cheapest_d_states[n]
        #cheapest_d_states_all[unique_periods] = cheapest_d_states[]
        
        # Report original demand levels and their total (TOU+seasonal) cost, for reference bill calculation
        
        # columns [:-1] of cheapest_possible_demands are the achievable demand levels, column [-1] is the cost
        # d_max_vector is an hourly vector of the demand level of that period (to become a max constraint in the DP), which is cast into an 8760 for the year.
        cheapest_possible_demands[month,:] = cheapest_d_states[:-1]
        charges_for_cheapest_demands[month] = cheapest_d_states[-1]
        demand_max_profile[month_hours[month]:month_hours[month+1]] = d_max_vector 
    
    results = {'cheapest_possible_demands':cheapest_possible_demands,
               'charges_for_cheapest_demands':charges_for_cheapest_demands,
               'demand_max_profile':demand_max_profile}
    
    return results
    
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
    
#%%
def tiered_calc_vec(values, levels, prices):
    # Vectorized piecewise function calculator
    values = np.asarray(values)
    levels = np.asarray(levels)
    prices = np.asarray(prices)
    y = np.zeros(values.shape)
    
    # Tier 1
    y += ((values >= 0) & (values < levels[:][:][0])) * (values*prices[:][:][0])

    # Tiers 2 and beyond    
    for tier in np.arange(1,np.size(levels,0)):
        y += ((values >= levels[:][:][tier-1]) & (values < levels[:][:][tier])) * (
            ((values-levels[:][:][tier-1])*prices[:][:][tier]) + levels[:][:][tier-1]*prices[:][:][tier-1])  
    
    return y
    
#%%

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out    
    
#%%
def irr_calc(values):
    """
    Return the minimum Internal Rate of Return (IRR) within the range [-30%,+Inf].

    This is the "average" periodically compounded rate of return
    that gives a net present value of 0.0; for a more complete explanation,
    see Notes below.

    Parameters
    ----------
    values : array_like, shape(N,)
        Input cash flows per time period.  By convention, net "deposits"
        are negative and net "withdrawals" are positive.  Thus, for example,
        at least the first element of `values`, which represents the initial
        investment, will typically be negative.

    Returns
    -------
    out : float
        Internal Rate of Return for periodic input values.

    Notes
    -----

    """
    res = np.roots(values[::-1])
    # Find the root(s) between 0 and 1
    mask = (res.imag == 0) & (res.real > 0)
    res = res[mask].real
    if res.size == 0:
        return np.nan
    rate = 1.0/res - 1
    if sum(values)>0:
        rate = rate[rate>0] # Negative IRR is returned otherwise
    rate = rate[rate>-.3]
    if rate.size == 0:
        return np.nan
    if rate.size == 1:
        rate = rate.item()
    else: 
        rate = min(rate)
    return rate
    

    
#%% Analysis of a 2D design space for a single building
    
######################## Define Directories ###################################
load_profile_dir = "C:\Users\pgagnon\Desktop\dGen\S+S Project\Python_workspace\NYSERDA_workspace"
PV_profile_dir = "C:\Users\pgagnon\Desktop\dGen\S+S Project\Python_workspace\NYSERDA_workspace"

######################## Import Tariff ###################################
tariff = rate_import.Rate("Inputs\Input_rate_Dispatcher.xlsx") # Import a blank rate, which we will manipulate later 


#################### Set details of this analysis ########################
n_years = 2
inc = 25
pv_sizes = np.linspace(0, 1700, inc)
batt_powers = np.linspace(0, 1000, inc)
batt_ratio = 3
d_inc_n = 50 # Number of demand levels between original and lowest possible that will be explored
demand_charge = 20 # $/kW
pv_cost = 3000
batt_kW_cost = 800
batt_kWh_cost = 250
ITC = 0.5 # placeholder for present value of ITC and MACRS

class batt:
    eta = 0.95 # battery half-trip efficiency
    cap = 0
    power = 0

load_temp = pd.read_csv('large_office_profile.csv')
pv_temp = pd.read_csv('pv_profile.csv')

load_profile = np.array(load_temp)
pv_cf_profiles = np.array(pv_temp)

# Calculate mean peak demand for bld with no system
results_base = demFun.demand_calculator(load_profile, tariff)
mean_demand_base = results_base['period_maxs_unratcheted'].mean()

mean_demand_sys = np.zeros((inc,inc))
mean_demand_reduction = np.zeros((inc,inc))
annual_bill_savings = np.zeros((inc,inc))
system_cost = np.zeros((inc,inc))
irr = np.zeros((inc,inc))


for p, pv_size in enumerate(pv_sizes):
    for b, batt_power in enumerate(batt_powers):
        print p, b
        results_storage = np.zeros((n_years,12))
        results_pv_and_storage = np.zeros((n_years,12))
        batt.cap = batt_power * 3
        batt.power = batt_power
        
        for year in range(n_years):
            load_profile_year = load_profile[:,year]
            pv_profile_year = pv_cf_profiles[:,year] * pv_size
            load_and_pv_profile_year = load_profile_year - pv_profile_year
            cheapest_possible_demands = np.zeros((12,tariff.d_n+1), float)
            demand_max_profile = np.zeros(len(load_profile_year), float)
            
            results_pv_and_storage_year = calc_min_possible_demands(d_inc_n, load_and_pv_profile_year, batt, tariff)
            results_pv_and_storage[year,:] = results_pv_and_storage_year['cheapest_possible_demands'].reshape(tariff.d_n*12)

        mean_demand_sys[p,b] = results_pv_and_storage.mean()
        mean_demand_reduction[p,b] = mean_demand_base - results_pv_and_storage.mean()
        annual_bill_savings[p,b] = mean_demand_reduction[p,b]*demand_charge*12
        system_cost[p,b] = (pv_size*pv_cost + batt.cap*batt_kWh_cost + batt.power*batt_kW_cost)*(1-ITC)
        cf = np.zeros(25)
        cf[:] = annual_bill_savings[p,b]
        cf[0] = -system_cost[p,b]
        irr[p,b] = irr_calc(cf)

#%%
# Plot of mean peak demand reduction
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis['bottom'] = new_fixed_axis(loc='bottom', axes=par2, offset=(0,-40))
par2.set_xlim(0,np.max(batt_powers)/np.max(load_profile))
par2.axis["right"].toggle(all=True)

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Temperature")
p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")

par1.set_ylim(0, 4)
par2.set_ylim(1, 65)

#new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#par2.axis['left'] = new_fixed_axis(loc='left', axes=par2, offset=(-60,0))
##par2.set_xlim(0,np.max(batt_powers)/np.max(load_profile))

#par2.axis['left'].toggle(all=True)

#host.set_xlim(0,2)
#host.set_ylim(0,2)

host.set_xlabel('Battery Size (kW)')
host.set_ylabel('PV Size (kW)')
par1.set_xlabel('placeholder')
#par2.set_ylabel('placeholder2')

X, Y = np.meshgrid(batt_powers, pv_sizes) # PV is Y, Battery is X

host.contourf(X, Y, mean_demand_reduction)

#plt.colorbar()
#plt.grid(True)
#plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
#
#plt.xlabel('Battery Size (kW)', size=14)
#plt.title('Mean peak demand reduction')

#%%
plt.figure(1, figsize=(5,5))
plt.contourf(X, Y, mean_demand_reduction)
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Size (kW)', size=14)
plt.title('Mean Reduction in Monthly Peak Demand\nLarge Office Building, Peak Demand = %d kW\n1 MW PV = 0.22 Annual Energy Penetration' % np.max(load_profile))

#%%
# plot of the reduction beyond the simple addition of the individual batt and PV potential
reduction_above_sum_potential = np.zeros((inc,inc))
for p, pv_size in enumerate(pv_sizes):
    for b, batt_power in enumerate(batt_powers):
        reduction_above_sum_potential[p,b] = mean_demand_reduction[p,b] / (mean_demand_reduction[0,b]+mean_demand_reduction[p,0])
reduction_above_sum_potential[0,0] = 1.0
plt.figure(2, figsize=(5,5))
plt.contourf(X, Y, reduction_above_sum_potential)
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Size (kW)', size=14)
plt.title('S+S Cooperation Ratio = \n\n(demand reduction of combined solar+storage systems)\n---------------------------------------------------------------------------\n(reduction of solar)+(reduction of storage)')
#plt.title('S+S Cooperation Ratio:\n' r'$\frac{\mathrm{demand reduction of combined Solar+Storage systems}}{(reduction of solar)+(reduction of storage)}$',
#         fontsize=14)
#%%
# plot of IRR
plt.figure(3, figsize=(5,5))
plt.contourf(X, Y, irr)
plt.colorbar()
plt.grid(True)
plt.ylabel('PV Size (kW)', rotation=0, labelpad=80, size=14)
plt.xlabel('Battery Size (kW)', size=14)
plt.title('IRRs')