# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:26:58 2016

@author: pgagnon

To Do:
# rewrite with the variables expected by the calculator
- Have it filter or keep track of energy tier units
- I have not checked the 12x24 to 8760 conversion
- Add list of tariff attributes to the class definition, for reference
"""

import requests as req
import numpy as np


#%%
class Tariff:
    """Doc goes here
    Tariff Attributes:
    -urdb_id: id for utility rate database. US, not international.            
    -name: tariff name
    -utility: Name of utility this tariff is associated with
    -fixed_charge: Fixed monthly charge in $/mo.
    -peak_kW_capacity_max: The annual maximum kW of demand that a customer can have and still be on this tariff
    -peak_kW_capacity_min: The annula minimum kW of demand that a customer can have and still be on this tariff
    -kWh_useage_max: The maximum kWh of average monthly consumption that a customer can have and still be on this tariff
    -kWh_useage_min: The minimum kWh of average monthly consumption that a customer can have and still be on this tariff
    -sector: residential, commercial, or industrial
    -comments: comments from the urdb
    -description: tariff description from urdb
    -source: uri for the source of the tariff
    -uri: link the the urdb page
    -voltage_category: secondary, primary, transmission        
    -d_flat_exists: Boolean of whether there is a flat (not tou) demand charge component. Flat demand is also called monthly or seasonal demand. 
    -d_flat_n: Number of unique flat demand period constructions
    -d_flat_prices: The prices of each tier/period combination for flat demand. Rows are tiers, columns are periods.
    -d_flat_levels: The limit (total kW) of each of each tier/period combination for flat demand. Rows are tiers, columns are periods.
    -d_tou_exists = Boolean of whether there is a tou (not flat) demand charge component
    -d_n = Number of unique tou demand periods
    -d_tou_prices = The prices of each tier/period combination for tou demand. Rows are tiers, columns are periods.    
    -d_tou_levels = The limit (total kW) of each of each tier/period combination for tou demand. Rows are tiers, columns are periods.
    -e_exists = Boolean of whether there is a flat (not tou) demand charge component
    -e_tou_exists = Boolean of whether there is a flat (not tou) demand charge component
    -e_n = Number of unique energy periods
    -e_prices = The prices of each tier/period combination for flat demand. Rows are tiers, columns are periods.    
    -e_levels = The limit (total kWh) of each of each tier/period combination for energy. Rows are tiers, columns are periods.
    -e_wkday_12by24: 12 by 24 period definition for weekday energy. Rows are months, columns are hours.
    -e_wkend_12by24: 12 by 24 period definition for weekend energy. Rows are months, columns are hours.
    -d_wkday_12by24: 12 by 24 period definition for weekday energy. Rows are months, columns are hours.
    -d_wkend_12by24: 12 by 24 period definition for weekend energy. Rows are months, columns are hours.
    -d_tou_8760
    -e_tou_8760
    -e_prices_no_tier
    -e_max_difference  
    -self.month_index: 8760 vector of month numbers, for later use in the bill calculator
    """
        
    def __init__(self, urdb_id=None):
        
        # Attach an 8760 vector of month numbers, for later use in the bill calculator
        month_hours = np.array([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760], int)
        self.month_index = np.zeros(8760, int)
        for month, hours in enumerate(month_hours):
            self.month_index[month_hours[month-1]:hours] = month-1
            
            
        if urdb_id==None:
            # Default values for a blank tariff
            self.urdb_id = 'No urdb id given'               
            self.name = 'User defined tariff - no name specified'
            self.utility = 'User defined tariff - no name specified'
            self.fixed_charge = 0
            self.peak_kW_capacity_max = 1e99,
            self.peak_kW_capacity_min = 0,
            self.kWh_useage_max = 1e99,
            self.kWh_useage_min = 0,
            self.sector = 'No sector specified',
            self.comments = 'No comments given',
            self.description = 'No description given',
            self.source = 'No source given',
            self.uri = 'No uri given',
            self.voltage_category = 'No voltage category given'        
            
            
            ###################### Blank Flat Demand Structure ########################
            self.d_flat_exists = False
            self.d_flat_n = 0
            self.d_flat_prices = np.zeros([1, 1])     
            self.d_flat_levels = np.zeros([1, 1])
                
            
            #################### Blank Demand TOU Structure ###########################
            self.d_tou_exists = False
            self.d_n = 0
            self.d_tou_prices = np.zeros([1, 1])     
            self.d_tou_levels = np.zeros([1, 1])
            
            
            ######################## Blank Energy Structure ###########################
            self.e_exists = False
            self.e_tou_exists = False
            self.e_n = 0
            self.e_prices = np.zeros([1, 1])     
            self.e_levels = np.zeros([1, 1])
            
                
            ######################## Blank Schedules ###########################
            self.e_wkday_12by24 = np.zeros([12,24], int)
            self.e_wkend_12by24 = np.zeros([12,24], int)
            self.d_wkday_12by24 = np.zeros([12,24], int)
            self.d_wkend_12by24 = np.zeros([12,24], int)
            
            ################### Blank 12x24s as 8760s Schedule ########################
            self.d_tou_8760 = np.zeros(8760, int)
            self.e_tou_8760 = np.zeros(8760, int)
            
            
            ######################## Precalculations ######################################
            self.e_prices_no_tier = np.max(self.e_prices, 0) # simplification until something better is implemented
            self.e_max_difference = np.max(self.e_prices) - np.min(self.e_prices)

        #%%
        # If given a urdb_id input argument, obtain and reshape that tariff through the URDB API      
        else:
            input_params = {'version':3,
                        'format':'json',
                        'detail':'full',
                        'getpage':urdb_id, # not real: 57d0b2315457a3120ec5b286 real: 57bcd2b65457a3a67e540154
                        'api_key':'bg51RuoT2OD733xqu0ehRRZWUzBGvOJuN5xyRtB4'}
        
            r = req.get('http://api.openei.org/utility_rates?', params=input_params)
            
            tariff_original = r.json()['items'][0]
                
            if 'label' in tariff_original: self.urdb_id = tariff_original['label']
            else: self.urdb_id = 'No urdb id given'            
            
            if 'name' in tariff_original: self.name = tariff_original['name']
            else: self.name = 'No name specified'
                
            if 'utility' in tariff_original: self.utility = tariff_original['utility']
            else: self.utility = 'No utility specified'
                
            if 'fixedmonthlycharge' in tariff_original: self.fixed_charge = tariff_original['fixedmonthlycharge']
            else: self.fixed_charge = 0
                
            if 'peakkwcapacitymax' in tariff_original: self.peak_kW_capacity_max = tariff_original['peakkwcapacitymax']
            else: self.peak_kW_capacity_max = 1e99
                
            if 'peakkwcapacitymin' in tariff_original: self.peak_kW_capacity_min = tariff_original['peakkwcapacitymin']
            else: self.peak_kW_capacity_min = 0
                
            if 'peakkwhusagemax' in tariff_original: self.kWh_useage_max = tariff_original['peakkwhusagemax']
            else: self.kWh_useage_max = 1e99
                
            if 'peakkwhusagemin' in tariff_original: self.kWh_useage_min = tariff_original['peakkwhusagemin']
            else: self.kWh_useage_min = 0
                
            if 'sector' in tariff_original: self.sector = tariff_original['sector']
            else: self.sector = 'No sector given'
                
            if 'basicinformationcomments' in tariff_original: self.comments = tariff_original['basicinformationcomments']
            else: self.comments = 'No comments'

            if 'description' in tariff_original: self.description = tariff_original['description']
            else: self.description = 'No description'
                
            if 'source' in tariff_original: self.source = tariff_original['source']
            else: self.source = 'No source given'

            if 'uri' in tariff_original: self.uri = tariff_original['uri']
            else: self.uri = 'No uri given'
                
            if 'voltage_category' in tariff_original: self.voltage_category = tariff_original['voltage_category']
            else: self.voltage_category = 'No voltage category given'
                 
            
            ###################### Repackage Flat Demand Structure ########################
            if 'flatdemandstructure' in tariff_original:
                self.d_flat_exists = True
                d_flat_structure = tariff_original['flatdemandstructure']
                self.d_flat_n = len(np.unique(tariff_original['flatdemandmonths']))
                
                # Determine the maximum number of tiers in the demand structure
                max_tiers = 1
                for period in range(self.d_flat_n):
                    n_tiers = len(d_flat_structure[period])
                    if n_tiers > max_tiers: max_tiers = n_tiers
                
                # Repackage Energy TOU Structure   
                self.d_flat_prices = np.zeros([max_tiers, self.d_flat_n], int)     
                self.d_flat_levels = np.zeros([max_tiers, self.d_flat_n], int)
                self.d_flat_levels[:,:] = 1e9
                for period in range(self.d_flat_n):
                    for tier in range(len(d_flat_structure[period])):
                        self.d_flat_levels[tier, period] = d_flat_structure[period][tier].get('max', 1e9)
                        self.d_flat_prices[tier, period] = d_flat_structure[period][tier].get('rate', 0) + d_flat_structure[period][tier].get('adj', 0)
            else:
                self.d_flat_exists = False
                self.d_flat_n = 0
                self.d_flat_prices = np.zeros([1, 1], int)     
                self.d_flat_levels = np.zeros([1, 1], int)
                
            
            #################### Repackage Demand TOU Structure ###########################
            if 'demandratestructure' in tariff_original:
                demand_structure = tariff_original['demandratestructure']
                self.d_n = len(demand_structure)
                if self.d_n > 1: self.d_tou_exists = True
                else: 
                    self.d_tou_exists = False
                    self.d_flat_exists = True
                
                # Determine the maximum number of tiers in the demand structure
                max_tiers = 1
                for period in range(self.d_n):
                    n_tiers = len(demand_structure[period])
                    if n_tiers > max_tiers: max_tiers = n_tiers
                
                # Repackage Demand TOU Structure   
                self.d_tou_prices = np.zeros([max_tiers, self.d_n], int)     
                self.d_tou_levels = np.zeros([max_tiers, self.d_n], int)
                self.d_tou_levels[:,:] = 1e9
                for period in range(self.d_n):
                    for tier in range(len(demand_structure[period])):
                        self.d_tou_levels[tier, period] = demand_structure[period][tier].get('max', 1e9)
                        self.d_tou_prices[tier, period] = demand_structure[period][tier].get('rate', 0) + demand_structure[period][tier].get('adj', 0)
            else:
                self.d_tou_exists = False
                self.d_n = 0
                self.d_tou_prices = np.zeros([1, 1], int)     
                self.d_tou_levels = np.zeros([1, 1], int)
            
            
            ######################## Repackage Energy Structure ###########################
            if 'energyratestructure' in tariff_original:
                self.e_exists = True
                energy_structure = tariff_original['energyratestructure']
                self.e_n = len(energy_structure)
                if self.e_n > 1: self.e_tou_exists = True
                else: self.e_tou_exists = False
                
                # Determine the maximum number of tiers in the demand structure
                max_tiers = 1
                for period in range(self.e_n):
                    n_tiers = len(energy_structure[period])
                    if n_tiers > max_tiers: max_tiers = n_tiers
                
                # Repackage Energy TOU Structure   
                self.e_prices = np.zeros([max_tiers, self.e_n], int)     
                self.e_levels = np.zeros([max_tiers, self.e_n], int)
                self.e_levels[:,:] = 1e9
                for period in range(self.e_n):
                    for tier in range(len(energy_structure[period])):
                        self.e_levels[tier, period] = energy_structure[period][tier].get('max', 1e9)
                        self.e_prices[tier, period] = energy_structure[period][tier].get('rate', 0) + energy_structure[period][tier].get('adj', 0)
            else:
                self.e_exists = False
                self.e_tou_exists = False
                self.e_n = 0
                self.e_prices = np.zeros([1, 1], int)     
                self.e_levels = np.zeros([1, 1], int)
                
            ######################## Repackage Energy Schedule ###########################
            self.e_wkday_12by24 = np.zeros([12,24], int)
            self.e_wkend_12by24 = np.zeros([12,24], int)
            
            if 'energyweekdayschedule' in tariff_original:
                for month in range(12):
                    self.e_wkday_12by24[month, :] = tariff_original['energyweekdayschedule'][month]
                    self.e_wkend_12by24[month, :] = tariff_original['energyweekendschedule'][month]
            
            ######################## Repackage Demand Schedule ###########################
            self.d_wkday_12by24 = np.zeros([12,24], int)
            self.d_wkend_12by24 = np.zeros([12,24], int)
            
            if 'demandweekdayschedule' in tariff_original:
                for month in range(12):
                    self.d_wkday_12by24[month, :] = tariff_original['demandweekdayschedule'][month]
                    self.d_wkend_12by24[month, :] = tariff_original['demandweekendschedule'][month]
            
            ################### Repackage 12x24s as 8760s Schedule ########################
            self.d_tou_8760 = np.zeros(8760, int)
            self.e_tou_8760 = np.zeros(8760, int)
            month = 0
            hour = 0
            day = 0
            for h in range(8760):
                if day < 5:
                    self.d_tou_8760[h] = self.d_wkday_12by24[month, hour]
                    self.e_tou_8760[h] = self.e_wkday_12by24[month, hour]
                else:
                    self.d_tou_8760[h] = self.d_wkend_12by24[month, hour]
                    self.e_tou_8760[h] = self.e_wkend_12by24[month, hour]
                hour += 1
                if hour == 24: hour = 0; day += 1
                if day == 7: day = 0
                if h > month_hours[month+1]: month += 1
            
            ######################## Precalculations ######################################
            self.e_prices_no_tier = np.max(self.e_prices, 0) # simplification until something better is implemented
            self.e_max_difference = np.max(self.e_prices) - np.min(self.e_prices)

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

def bill_calculator(load_profile, tariff, export_tariff):
    """
    Not vectorized for now. Next step will be pass in multiple profiles for the same tariff
    For now, only 2 styles of NEM: Full retail and fixed schedule value. 

    To-do
    -Both energy and demand calcs (anything that uses piecewise calculator) doesn't go below zero, because piecewise isn't built to.
        Therefore, credit can't be earned in one period and applied to another. 
    -My current approach sums for all periods, not just those in a month. Potentially inefficient, if it was dynamic it would be cheaper, but less clear.
    -Make this flexible for different hours increments (which will require more robust approach for power vs energy units)
    -Not sure what happens if there is no energy component in the tariff, at the moment    
    """
    
    n_months = 12
    n_timesteps = 8760
    if len(tariff.d_tou_8760) != 8760: 
        print 'Warning: Non-8760 profiles are not yet supported by the bill calculator'
    
    ###################### Calculate TOU Demand Charges ###########################
    if tariff.d_tou_exists == True:
        # Cast the TOU periods into a boolean matrix
        period_matrix = np.zeros([n_timesteps, tariff.d_n*n_months], bool)
        period_matrix[range(n_timesteps),tariff.d_tou_8760+tariff.month_index*tariff.d_n] = True
        
        # Determine the max demand in each period of each month of each year
        load_distributed = load_profile[np.newaxis, :].T*period_matrix
        period_maxs = np.max(load_distributed, axis=0)
        
        # Calculate the cost of TOU demand charges
        d_TOU_period_costs = tiered_calc_vec(period_maxs, np.tile(tariff.d_tou_levels[:,0:tariff.d_n], 12), np.tile(tariff.d_tou_prices[:,0:tariff.d_n], 12))
       
        d_TOU_month_totals = np.zeros([n_months])
        for month in range(n_months):
            d_TOU_month_totals[month] = np.sum(d_TOU_period_costs[(month*tariff.d_n):(month*tariff.d_n + tariff.d_n)])
    else:
        d_TOU_month_totals = np.zeros([n_months])
        
    ################ Calculate Flat Demand Charges ############################
    if tariff.d_flat_exists == True:
        # Cast the seasons into a boolean matrix
        flat_matrix = np.zeros([n_timesteps, n_months], bool)
        flat_matrix[range(n_timesteps),tariff.month_index] = True
        
        # Determine the max demand in each month of each year
        load_distributed = load_profile[np.newaxis, :].T*flat_matrix
        flat_maxs = np.max(load_distributed, axis=0)
        
        flat_costs = tiered_calc_vec(flat_maxs, tariff.d_flat_levels, tariff.d_flat_prices)  
    else:
        flat_costs = np.zeros([n_months])
    
    ################ Calculate Energy Charges ############################
    if export_tariff.full_retail_nem == False:
        imported_profile = np.clip(load_profile, 0, 1e99)
        exported_profile = np.clip(load_profile, -1e99, 0)

        # Calculate fixed schedule export_tariff 
        # Cast the TOU periods into a boolean matrix
        export_period_matrix = np.zeros([len(export_tariff.periods_8760), export_tariff.period_n*n_months], bool)
        export_period_matrix[range(len(export_tariff.periods_8760)),export_tariff.periods_8760+tariff.month_index*export_tariff.period_n] = True
        
        # Determine the energy consumed in each period of each month of each year
        load_distributed = exported_profile[np.newaxis, :].T*export_period_matrix
        export_period_sums = np.sum(load_distributed, axis=0)
        
        # Calculate the cost of TOU demand charges
        export_period_credits = tiered_calc_vec(export_period_sums, np.tile(export_tariff.levels[:,0:export_tariff.period_n], 12), np.tile(export_tariff.prices[:,0:export_tariff.period_n], 12))
        
        export_month_totals = np.zeros([n_months])
        for month in range(n_months):
            export_month_totals[month] = np.sum(export_period_credits[(month*export_tariff.period_n):(month*export_tariff.period_n + export_tariff.period_n)])        
            
        # Calculate imported energy charges. 
        # Cast the TOU periods into a boolean matrix
        e_period_matrix = np.zeros([len(tariff.e_tou_8760), tariff.e_n*n_months], bool)
        e_period_matrix[range(len(tariff.e_tou_8760)),tariff.e_tou_8760+tariff.month_index*tariff.e_n] = True
        
        # Determine the max demand in each period of each month of each year
        load_distributed = imported_profile[np.newaxis, :].T*e_period_matrix
        e_period_sums = np.sum(load_distributed, axis=0)
        
        # Calculate the cost of TOU demand charges
        e_period_costs = tiered_calc_vec(e_period_sums, np.tile(tariff.e_levels[:,0:tariff.e_n], 12), np.tile(tariff.e_prices[:,0:tariff.e_n], 12))
        
        e_month_totals = np.zeros([n_months])
        for month in range(n_months):
            e_month_totals[month] = np.sum(e_period_costs[(month*tariff.e_n):(month*tariff.e_n + tariff.e_n)])
            
        net_e_month_totals = e_month_totals - export_month_totals
      
    else:
        # Calculate imported energy charges with full retail NEM
        # Cast the TOU periods into a boolean matrix
        e_period_matrix = np.zeros([len(tariff.e_tou_8760), tariff.e_n*n_months], bool)
        e_period_matrix[range(len(tariff.e_tou_8760)),tariff.e_tou_8760+tariff.month_index*tariff.e_n] = True
        
        # Determine the energy consumed in each period of each month of each year netting exported electricity
        load_distributed = load_profile[np.newaxis, :].T*e_period_matrix
        e_period_sums = np.sum(load_distributed, axis=0)
        
        # Calculate the cost of TOU energy charges netting exported electricity
        e_period_costs = tiered_calc_vec(e_period_sums, np.tile(tariff.e_levels[:,0:tariff.e_n], 12), np.tile(tariff.e_prices[:,0:tariff.e_n], 12))
        
        net_e_month_totals = np.zeros([n_months])
        for month in range(n_months):
            net_e_month_totals[month] = np.sum(e_period_costs[(month*tariff.e_n):(month*tariff.e_n + tariff.e_n)])
        
        # Determine the value of NEM
        # Calculate imported energy charges with zero exported electricity
        imported_profile = np.clip(load_profile, 0, 1e99)
        exported_profile = np.clip(load_profile, -1e99, 0)

        # Determine the energy consumed in each period of each month of each year - without exported electricity
        load_distributed = imported_profile[np.newaxis, :].T*e_period_matrix
        ref_e_period_sums = np.sum(load_distributed, axis=0)
        
        # Calculate the cost of TOU energy charges without exported electricity
        ref_e_period_costs = tiered_calc_vec(ref_e_period_sums, np.tile(tariff.e_levels[:,0:tariff.e_n], 12), np.tile(tariff.e_prices[:,0:tariff.e_n], 12))
        
        ref_e_month_totals = np.zeros([n_months])
        for month in range(n_months):
            ref_e_month_totals[month] = np.sum(ref_e_period_costs[(month*tariff.e_n):(month*tariff.e_n + tariff.e_n)])
        
        # Determine how much  the exported electricity was worth by comparing
        # bills where it was netted against those where it wasn't
        export_month_totals = net_e_month_totals - ref_e_month_totals
        
    total_monthly_bills = d_TOU_month_totals + flat_costs + net_e_month_totals + tariff.fixed_charge
    annual_bill = sum(total_monthly_bills)
        
    tariff_results = {'annual_bill':annual_bill,
                    'total_monthly_bills':total_monthly_bills,
                    'monthly_d_charges':d_TOU_month_totals + flat_costs,
                    'net_e_month_totals':net_e_month_totals,
                    'credit_from_export':export_month_totals,
                    'e_period_charges':e_period_costs,
                    'e_period_sums':e_period_sums}
    
    return annual_bill, tariff_results
    

#%%

blank_t = Tariff()
real_t = Tariff('57bcd2b65457a3a67e540154')

class export_tariff:
    """
    Structure of compensation for exported generation. Currently only two 
    styles: full-retail NEM, and instantanous TOU energy value. 
    """
     
    full_retail_nem = False
    prices = np.zeros([1, 1], int)     
    levels = np.zeros([1, 1], int)
    periods_8760 = np.zeros(8760, int)
    period_n = 1

load_profile = np.random.normal(100, 10, 8760)
bill, results = bill_calculator(load_profile, real_t, export_tariff)
                