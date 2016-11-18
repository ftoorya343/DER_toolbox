# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:07:47 2016

@author: pgagnon
"""

import numpy as np
from support_functions import tiered_calc_vec, cartesian
import pandas as pd


def demand_calculator(load_profile, r):

    n_periods = np.max(r.tou_schedule)+1 # Number of tou periods in this rate
    n_years = np.size(load_profile,1) # Number of years in the load profile
    n_months = 12
    profile_len = int(np.size(load_profile, 0))
    month_index = np.zeros(profile_len, int)
    month_hours = np.array([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]) * profile_len / 8760;
    for month, hours in enumerate(month_hours):
        month_index[month_hours[month-1]:hours] = month-1
    
    ###################### Calculate TOU Demand Charges ###########################
    # Cast the TOU periods into a boolean matrix
    period_matrix = np.zeros([len(r.tou_schedule), n_periods*n_months], bool)
    period_matrix[range(len(r.tou_schedule)),r.tou_schedule+month_index*n_periods] = True
    
    # Determine the max demand in each period of each month of each year
    period_maxs = np.zeros([n_years, n_periods*n_months])
    for year in range(n_years):
        load = load_profile[:,year]
        load_distributed = load[np.newaxis, :].T*period_matrix
        period_maxs[year,:] = np.max(load_distributed, axis=0)
    
    # Apply ratchet (currently only for TOU charges, but could expand to seasonal as well)
    period_demand_matrix = np.zeros([n_periods, n_months*n_years])
    for period in range(n_periods):
        # Recast into matrix where each row is a period, and the columns are all of the max demands for each month of each year
        period_demand_matrix[period,:] = np.reshape(period_maxs[:,np.arange(0,n_months)*n_periods+period], n_years*n_months)
    
    period_demand_ratcheted = np.zeros([n_periods, n_months*n_years])
    historical_demands = np.zeros([n_periods, int(r.ratchet_memory+1)])
    period_maxs_ratcheted = np.zeros([n_years, n_periods*n_months])
    for month in range(np.size(period_demand_matrix,1)):
        # Step through each column (month) of the period_demand_matrix
        # For each month, obtain the set of historical max demand values reduced by ratchet_fraction, and current month's max demand
        historical_demands[:,:-1] = period_demand_matrix[:,np.arange(month-r.ratchet_memory,month)]*r.ratchet_fraction
        historical_demands[:,-1] = period_demand_matrix[:,month]
        # Identify the demand level to use for demand charge calculation
        period_demand_ratcheted[:,month] = np.max(historical_demands, 1)
        # Recast into original matrix structure
        #period_maxs_ratcheted[np.floor(month/n_months),np.mod(month*n_months,24):np.mod(month*n_months+n_periods,24)] = np.max(historical_demands, 1)
        period_maxs_ratcheted[int(np.floor(month/n_months)),np.mod(month*n_periods,n_months*n_periods):np.mod(month*n_periods,n_months*n_periods)+n_periods] = np.max(historical_demands, 1)

    # Calculate the cost of TOU demand charges
    TOU_period_costs = tiered_calc_vec(period_maxs_ratcheted, np.tile(r.tou_levels[:,0:n_periods], 12), np.tile(r.tou_prices[:,0:n_periods], 12))
    TOU_period_costs_no_ratchet = tiered_calc_vec(period_maxs, np.tile(r.tou_levels[:,0:n_periods], 12), np.tile(r.tou_prices[:,0:n_periods], 12))
    
    TOU_month_totals = np.zeros([n_years, n_months])
    for month in range(n_months):
        TOU_month_totals[:,month] = np.sum(TOU_period_costs[:,(month*n_periods):(month*n_periods + n_periods)], 1)
        
    ######### Calculate Seasonal (monthly) Demand Charges #####################
    # Cast the seasons into a boolean matrix
    seasonal_matrix = np.zeros([len(r.tou_schedule), 12], bool)
    seasonal_matrix[range(len(r.tou_schedule)),month_index] = True
    
    # Determine the max demand in each month of each year
    seasonal_maxs = np.zeros([n_years, n_months])
    max_indicies = np.zeros([n_years, n_months], int)

    for year in range(n_years):
        load = load_profile[:,year]
        load_distributed = load[np.newaxis, :].T*seasonal_matrix
        seasonal_maxs[year,:] = np.max(load_distributed, axis=0)
        
        # The index of the year in which each month's max occured
        max_indicies[year,:] = np.argmax(load_distributed, axis=0)
    
    seasonal_costs = tiered_calc_vec(seasonal_maxs, r.seasonal_levels, r.seasonal_prices)

    ######### Identify day/hour of peak demand for each month ##############
    # The day of the month in which the peak demand of that month occured    
    peak_days = np.floor((max_indicies - month_hours[:-1]) / 48.0)
    
    # The hour of the day in which the peak demand of that month occured
    peak_hours = np.mod((max_indicies - month_hours[:-1]) / 48.0, 1) * 24.0
    

    
    ############### Package Results ###############################################
    total_annual_demand_charges = np.sum(TOU_month_totals,1) + np.sum(seasonal_costs,1)
    monthly_totals = TOU_month_totals + seasonal_costs
    ratchet_costs = np.sum(TOU_period_costs,1) - np.sum(TOU_period_costs_no_ratchet,1)
    
    annual_peak_i = np.argmax(load_profile, axis=0)
    
    
    results = {'ratchet_costs':ratchet_costs,
                    'annual_peak_i':annual_peak_i,
                    'period_maxs_ratcheted':period_maxs_ratcheted,
                    'period_maxs_unratcheted':period_maxs,
                    'total_annual_demand_charges':total_annual_demand_charges,
                    'monthly_totals':monthly_totals,
                    'peak_days':peak_days,
                    'peak_hours':peak_hours}
    
    # All results are packaged into a dictionary, for flexible extraction
    return results
    
    # Other potential outputs:
    # How much the ratchet cost
    # Hours when demands charges were set
    # Average charges over the years
    # Raw demand values (to determine capacity credits)
    # Either bring in PV profile here and output capacity values, or keep this a 
    #   rate calculator and have that process performed in a higher level function
    

#%%
def import_load_profiles_for_loc(load_profile_dir, loc, load_profiles, currently_loaded_loc):
    # Import the building load profile csv into a pandas dataframe
    # Since this is computationally expensive, we check to see if we've already
    # loaded it for this location, and skip if we already have.

    if currently_loaded_loc != loc:
        load_profiles = pd.read_csv(load_profile_dir + '\%s.csv' % loc)
        currently_loaded_loc = loc
        
        
        # patching missing building types
        if loc == 'HOUSTON':
            load_profiles['RefBldgFullServiceRestaurantNew2004%s' % loc] = load_profiles['RefBldgFullServiceRestaurantPost1980%s' % loc].copy()
        if loc == 'SAN_FRANCISCO':
            load_profiles['RefBldgWarehousePre1980%s' % loc] = load_profiles['RefBldgWarehouseNew2004%s' % loc].copy()
            load_profiles['RefBldgWarehousePost1980%s' % loc] = load_profiles['RefBldgWarehouseNew2004%s' % loc].copy()
            load_profiles['RefBldgQuickServiceRestaurantPost1980%s' % loc] = load_profiles['RefBldgQuickServiceRestaurantNew2004%s' % loc].copy()
            load_profiles['RefBldgQuickServiceRestaurantPre1980%s' % loc] = load_profiles['RefBldgQuickServiceRestaurantNew2004%s' % loc].copy()
            load_profiles['RefBldgStand-aloneRetailNew2004%s' % loc] = load_profiles['RefBldgStand-aloneRetailPost1980%s' % loc].copy()
            load_profiles['RefBldgStripMallNew2004%s' % loc] = load_profiles['RefBldgStripMallPost1980%s' % loc].copy()
            load_profiles['RefBldgSmallOfficePost1980%s' % loc] = load_profiles['RefBldgSmallOfficeNew2004%s' % loc].copy()
        
        load_profiles['RefBldgQuickServiceRestaurantPre1980%s' % loc] = load_profiles['RefBldgQuickServiceRestaurantPost1980%s' % loc].copy() 
        load_profiles['RefBldgStand-aloneRetailNew2004%s' % loc] = load_profiles['RefBldgStand-aloneRetailPost1980%s' % loc].copy()
        


    return load_profiles, currently_loaded_loc

#%%

# Only use for demand charge project load profiles
def select_and_clean_load_profile(load_profiles, loc, bld):
    # Purpose: Select the relevent building load profile for the location of interest, from a dataframe of load profiles
    
    # Clean
    load_profiles.loc[0,'%s%s' %(bld, loc)] = 0
    load_profiles = load_profiles[load_profiles['%s%s' %(bld, loc)] > -1000]
    load_profiles.drop(load_profiles.index[-1:], inplace=True)
    
    # Scale by two, to get into kW, from kWh
    org_load_profile = np.array(load_profiles['%s%s' %(bld, loc)]).reshape([17520,17], order='F') * 2
    
    # Check to make sure input profile is proper length
    profile_res = 60*8760/int(np.size(org_load_profile, 0))
    if profile_res != 15 and profile_res != 30 and profile_res != 60:
        print "Timesteps other than 60, 30, or 15 minutes are not currently supported"
    
    return org_load_profile, profile_res
    
    
#%%
def adjust_resolutions(org_load_profile, r):
    '''
    Purpose: Adjusts either the load profile resolution or the rate's demand
            window resolution, depending on which is smaller. Also expands the
            rate's period schedule array, if necessary.     
    '''
    
    
    profile_res = 60*8760/int(np.size(org_load_profile, 0))
    
    # adjusting as necessary
    if r.window < profile_res:
        #print "Demand window is smaller than input load profile resolution."
        #print "Running analysis with demand window set to load profile resolution"
        r.window = profile_res
        load_profile = org_load_profile
    elif r.window > profile_res:
        #print "Input load profile is finer resolution than demand window."
        #print "Collapsing load profile to window timesteps by averaging."
        load_profile = np.zeros([8760*60/r.window, np.size(org_load_profile, 1)])
        i_for_collapsing = np.arange(0,np.size(org_load_profile, 0)-1,r.window/profile_res, int)
        for n in range(int(r.window/profile_res)):
            load_profile += org_load_profile[i_for_collapsing+n,:]
        load_profile = load_profile / (r.window/profile_res)
    else: 
        #print "Load profile resolution equals demand window, no changes made."
        load_profile = org_load_profile
    
    profile_len = int(np.size(load_profile, 0))
    
    # Expanding period schedule if necessary
    # This assumes that schedules will still be defined hourly, even if metering is sub-hourly
    if len(r.tou_schedule) != profile_len:
        org_tou_schedule = r.tou_schedule
        r.tou_schedule = np.zeros(profile_len, int)
        for n in range(profile_len/8760):
            r.tou_schedule[np.arange(0+n,profile_len,profile_len/8760)] = org_tou_schedule
            
    return load_profile, profile_len, r
    
#%%
# Only use this for demand project profiles
def import_pv_cf_profile(PV_profile_dir, loc, tilt, azimuth, profile_len, pv_cf_profiles, currently_loaded_pv):
    '''
    Purpose: Imports the PV productivity profiles. Because they were generated 
            for a 100 kW system, they are divided by 100 to represent the
            normalized capacity factor.
    '''
    pv_loc_codes = {'ALBUQUERQUE':431499, 'ATLANTA':956302, 'BALTIMORE':1144401, 
                'BOULDER':464388, 'CHICAGO':884454, 'DULUTH':787594, 
                'HELENA':311969, 'HOUSTON':721531, 'LAS_VEGAS':247402, 
                'LOS_ANGELES':187962, 'MIAMI':1045605, 'MINNEAPOLIS':766157,
                'PHOENIX':311167, 'SAN_FRANCISCO':124782, 'SEATTLE':125711}    
    
    if currently_loaded_pv != 'tilt%s_azimuth%s_%d.csv' %(tilt, azimuth, pv_loc_codes[loc]):
        pv_cf_profiles = np.zeros([17520, 17])
        for n, year in enumerate(np.arange(1998,2015,1)):
            file_string = PV_profile_dir + '\PV_prod_tilt%s_azimuth%s' %(tilt, azimuth) + '/tilt%s_azimuth%s_%d-%d.csv' %(tilt, azimuth, pv_loc_codes[loc], year)
            pv_cf_profiles[:,n] = np.genfromtxt(file_string) / 100 * profile_len/8760 # /100 because 100kW system in simulation, then scale because its kWh not kW
        
        currently_loaded_pv =  'tilt%s_azimuth%s_%d.csv' %(tilt, azimuth, pv_loc_codes[loc])      
        
    return pv_cf_profiles, currently_loaded_pv
    
#%%
# This function is no longer used - pgagnon 5/17/16
def create_building_list(bld_types, vints, locs):
    # Builds a list of building-vintages (for iterating)
    # Also builds a building-vintage-location list (for indexing)
    blds = list()
    bld_loc_list = list()
    for bld_type in bld_types:
        for vint in vints:
            if bld_type == 'QuickServiceRestaurant' and vint == 'Pre1980': pass
            if bld_type == 'Stand-aloneRetail' and vint == 'New2004': pass
            blds.append('RefBldg%s%s' %(bld_type, vint))
            for loc in locs:
                bld_loc_list.append('RefBldg%s%s%s' %(bld_type, vint, loc))

    return blds, bld_loc_list
    
#%%
    
def initialize_agent_df(bld_types, vints, locs, pv_azimuths, pv_slopes, pv_fractions):
    '''
    Script: initialize_agent_df.py
    Purpose: Builds a datarame of agents and their attirbutes that contains 
    all the combinatorial possabilities of the input attributes. 
    
    Inputs: 6 lists of attributes
        Each attribute must always have at least one value specified
    
    Outputs: agent_df (pandas dataframe) 

    '''

    bld_loc_list = list()
    agent_list = list()

    # Generate a matrix of the combinatorial possabilities of the given building attributes
    agent_matrix = np.array(cartesian([bld_types, vints, locs, pv_azimuths, pv_slopes, pv_fractions]), dtype='|S22')
    agent_matrix[agent_matrix=='SAN_FRANCIS'] = 'SAN_FRANCISCO'
    
    for row in range(np.size(agent_matrix, 0)):
        bld_type = agent_matrix[row,0]
        vint = agent_matrix[row,1]
        loc = agent_matrix[row,2]
        pv_azimuth = agent_matrix[row,3]
        pv_slope = agent_matrix[row,4]
        pv_fraction = agent_matrix[row,5]
        
        # This is a temporary patch, because of several missing profile types
        #if bld_type == 'QuickServiceRestaurant' and vint == 'Pre1980': vint = 'Post1980'; agent_matrix[row,1] = vint = 'post1980'
        #if bld_type == 'Stand-aloneRetail' and vint == 'New2004': vint = 'post1980'; agent_matrix[row,1] = vint = 'post1980'
        
        bld_loc_list.append('RefBldg%s%s' %(bld_type, vint))
        agent_list.append('%s%s%s_az%s_s%s_f%s' %(bld_type, vint, loc, pv_azimuth, pv_slope, pv_fraction))
        
    agent_df = pd.DataFrame(index = agent_list)
    agent_df['bld'] = bld_loc_list
    agent_df['bld_type'] = agent_matrix[:,0]
    agent_df['vint'] = agent_matrix[:,1]
    agent_df['loc'] = agent_matrix[:,2]
    
    azi_strings = agent_matrix[:,3]
    slope_strings = agent_matrix[:,4]
    pv_f_strings = agent_matrix[:,5]
    azis = [int(i) for i in azi_strings]
    slopes = [int(i) for i in slope_strings]
    pv_fs = [float(i) for i in pv_f_strings]
    
    agent_df['azi'] = azis
    agent_df['slope'] = slopes
    agent_df['pv_f'] = pv_fs # annual energy fraction supplied by PV
    
    # Sorts alphabetically by the location column, to minimize reloading location csv's later
    agent_df = agent_df.sort(columns='loc')

    return agent_df

#%%
    
def initialize_agent_df_res(heats, foundations, codes, locs, pv_azimuths, pv_slopes, pv_fractions):
    '''
    Script: initialize_agent_df.py
    Purpose: Builds a datarame of agents and their attirbutes that contains 
    all the combinatorial possabilities of the input attributes. 
    
    Inputs:
    
    Outputs: agent_df (pandas dataframe) 

    '''

    bld_loc_list = list()
    agent_list = list()

    # Generate a matrix of the combinatorial possabilities of the given building attributes
    agent_matrix = np.array(cartesian([heats, foundations, codes, locs, pv_azimuths, pv_slopes, pv_fractions]), dtype='|S22')
    #agent_matrix[agent_matrix=='SAN_FRANCIS'] = 'SAN_FRANCISCO'
    
    for row in range(np.size(agent_matrix, 0)):
        heat = agent_matrix[row,0]
        foundation = agent_matrix[row,1]
        code = agent_matrix[row,2]
        loc = agent_matrix[row,3]
        pv_azimuth = agent_matrix[row,4]
        pv_slope = agent_matrix[row,5]
        pv_fraction = agent_matrix[row,6]
        
        bld_loc_list.append('RefBldg%s%s' %(bld_type, vint))
        agent_list.append('%s%s%s_az%s_s%s_f%s' %(bld_type, vint, loc, pv_azimuth, pv_slope, pv_fraction))
        
    agent_df = pd.DataFrame(index = agent_list)
    agent_df['bld'] = bld_loc_list
    agent_df['bld_type'] = agent_matrix[:,0]
    agent_df['vint'] = agent_matrix[:,1]
    agent_df['loc'] = agent_matrix[:,2]
    
    azi_strings = agent_matrix[:,3]
    slope_strings = agent_matrix[:,4]
    pv_f_strings = agent_matrix[:,5]
    azis = [int(i) for i in azi_strings]
    slopes = [int(i) for i in slope_strings]
    pv_fs = [float(i) for i in pv_f_strings]
    
    agent_df['azi'] = azis
    agent_df['slope'] = slopes
    agent_df['pv_f'] = pv_fs # annual energy fraction supplied by PV
    
    # Sorts alphabetically by the location column, to minimize reloading location csv's later
    agent_df = agent_df.sort(columns='loc')

    return agent_df

    
    
    