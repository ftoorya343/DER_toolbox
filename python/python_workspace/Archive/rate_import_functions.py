# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:55:13 2016

@author: pgagnon
"""

# Imports rate structure from excel input sheet

import openpyxl as xl
from named_range_functions import FancyNamedRange
import pandas as pd
import numpy as np

class Rate(object):
    
    def __init__(self, workbook):
        xls_file = workbook
        wb = xl.load_workbook(xls_file, data_only = True) # open the workbook         

        ################ Import Seasonal Demand Charge Structure ##############
        fnr = FancyNamedRange(wb, "seasonal_levels")
        self.seasonal_levels = np.array(fnr.data_frame, float)
        
        fnr = FancyNamedRange(wb, "seasonal_prices")
        self.seasonal_prices = np.array(fnr.data_frame, float)
        
        ####################### Import Ratchet Structure ##############
        fnr = FancyNamedRange(wb, "ratchet_fraction")
        self.ratchet_fraction = np.array(fnr.data_frame, float)
        
        fnr = FancyNamedRange(wb, "ratchet_memory")
        self.ratchet_memory = np.array(fnr.data_frame, int)
        
        ##################### Import TOU Demand Charge Structure ##############
        fnr = FancyNamedRange(wb, "tou_weekday_schedule")
        tou_weekday_schedule = np.array(fnr.data_frame, int)
        self.tou_weekday_schedule = np.array(fnr.data_frame, int)
        
        fnr = FancyNamedRange(wb, "tou_weekend_schedule")
        tou_weekend_schedule = np.array(fnr.data_frame, int)
        self.tou_weekend_schedule = np.array(fnr.data_frame, int)
        
        self.d_n = int(np.max(tou_weekday_schedule))
        
        fnr = FancyNamedRange(wb, "tou_levels")
        self.tou_levels = np.array(fnr.data_frame, float)
        
        fnr = FancyNamedRange(wb, "tou_prices")
        self.tou_prices = np.array(fnr.data_frame, float)

        fnr = FancyNamedRange(wb, "demand_window")
        self.window = np.array(fnr.data_frame.iloc[0,0], float)
        
        ######## Build 8760 TOU Period Vector out of 12x24 Schedule ###########
        
        month_hours =  [744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
        tou_schedule = np.zeros(8760)
        month = 0
        hour = 0
        day = 0
        for h in range(8760):
            if day < 5:
                tou_schedule[h] = tou_weekday_schedule[month, hour]
            else:
                tou_schedule[h] = tou_weekend_schedule[month, hour]
            hour += 1
            if hour == 24: hour = 0; day += 1
            if day == 7: day = 0
            if h > month_hours[month]: month += 1
        
        tou_schedule = tou_schedule - 1 # Offset by 1 for indexing
        self.tou_schedule = np.array(tou_schedule,int)
                
                
        ##################### Metadata ##############
        self.name = 'placeholder for rate names'
        self.utility = 'placeholder for utility names'
        self.ID = 'placerholder for rate identification number'
    
    def assign_tou_schedule(self, tou_weekday_schedule, tou_weekend_schedule, profile_len):
        # Accepts two 12x24 period matricies, reassigns them to an 8760
        month_hours =  [744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
        tou_schedule = np.zeros(8760)
        month = 0
        hour = 0
        day = 0
        for h in range(8760):
            if day < 5:
                tou_schedule[h] = tou_weekday_schedule[month, hour]
            else:
                tou_schedule[h] = tou_weekend_schedule[month, hour]
            hour += 1
            if hour == 24: hour = 0; day += 1
            if day == 7: day = 0
            if h > month_hours[month]: month += 1
        
        # Offset by 1 for indexing
        org_tou_schedule = tou_schedule - 1 
        
        # Adjust the resolution based on the resolution of the associated load profile
        self.tou_schedule = np.zeros(profile_len, int)
        for t in range(profile_len/8760):
            self.tou_schedule[np.arange(0+t,profile_len,profile_len/8760)] = org_tou_schedule
            
                