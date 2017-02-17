# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:02:02 2017

@author: pgagnon
"""

import tariff_functions as tFuncs
import dispatch_functions as dFuncs
import numpy as np
import pandas as pd
import csv

# In order to use certain tariff functions, you will need a URDB API key, which
# will then be ingested via the config file. Copy the config template from
# the templates_and_master_lists folder into your working directory, rename it,
# and enter your API key into the file. 
# You can get an API key here: http://en.openei.org/services/api/signup/
config = tFuncs.load_config_params('config.json')

#%%
# The download_tariffs_from_urdb does bulk downloads of tariffs. Currently it
# can either download all tariffs for a given sector, or for a given utility.
download_all_commercial_tariffs = False
download_tariffs_for_a_utility = True

if download_all_commercial_tariffs:
    tariff_df_sector = tFuncs.download_tariffs_from_urdb(config['api_key'], sector='Commercial')
    
if download_tariffs_for_a_utility:
    tariff_df = tFuncs.download_tariffs_from_urdb(config['api_key'], utility='Potomac Electric Power Co')


#%%
# The tariffs that are downloaded from the URDB are likely not all applicable
# for any given analysis. I have been collecting a set of keywords to filter
# by, to curate a smaller set of tariffs that represent what would likely be
# available for a "typical" commercial customer. Copy the 
# master_keyword_list_for_tariff_exclusion.csv file from the
# templates_and_master_lists folder into your working directory, rename it,
# and look through the list to determine which of them you may not want to
# filter by for your analysis.
keyword_list_file = 'keyword_list_for_tariff_exclusion.csv'

# filter_tariff_df applies the keyword filters to the tariff names, as well
# as filtering by the unit type and whether the tariff is expired.
included_tariffs, excluded_tariffs, keyword_count_df =  tFuncs.filter_tariff_df(tariff_df, 
                                                       keyword_list_file=keyword_list_file)
                                                       
# Important!!
# Note that the URDB is not perfect, and this filtering will not catch all
# bad tariffs. You should have some type of further validation, as appropriate
# for the specific analysis. E.g., divide the total bill by total energy
# consumption, and verify that the cents/kWh average is within a reasonable
# range.

                                                       
#%%
# The included_tariffs result only contains the tariff metadata. To do bill
# calculations with any of the tariffs, it will be necessary to use the Tariff
# class object to query the URDB API for that specific tariff.
example_urdb_id = included_tariffs.loc[list(included_tariffs.index)[0], 'label']
# or
example_urdb_id = '5785463a5457a3707529b89f'


# This creates the tariff class object.
tariff = tFuncs.Tariff(urdb_id = example_urdb_id, api_key=config['api_key'])

# Note that you can also save the tariff objects as a json file and ingest them
# from your local machine - this saves you from having to make repeated API 
# calls and makes sure that the tariff will always be available. 
# tariff = tFuncs.Tariff(json_file_name='file_path_for_tariff_on_disk.json')

# The compensation for exported generation is handled as a separate object. 
# Currently, the only options are full-retail-net-metering or an 8760 energy 
# price vector.  
export_tariff = tFuncs.Export_Tariff(full_retail_nem=True)

# Ingest your load profile however is appropriate.
load_profile = np.genfromtxt('example_load_profile_lg_office_denver.csv')

# Run the bill calculator
annual_bill, bill_results = tFuncs.bill_calculator(load_profile, tariff, export_tariff)
print "Total annual electric bill: $", annual_bill
print "Annual Demand Charges: $", bill_results['d_charges']
print "Annual Energy Charges: $", bill_results['e_charges']

#%%
# I also have developed a battery dispatcher, which seeks to minimize the 
# electric bill of the host, given an initial load profile and tariff.

# Create a battery object
batt = dFuncs.Battery(nameplate_cap=300.0, nameplate_power=100.0)

# The dispatcher accepts the pv_profile separately, since it has the option
# to restrict the battery to charge only from PV
pv_profile = np.zeros(8760)

# Perform the dispatch
dispatch_results = dFuncs.determine_optimal_dispatch(load_profile, pv_profile, batt, tariff, export_tariff)
annual_bill_dispatch, bill_results_dispatch = tFuncs.bill_calculator(dispatch_results['load_profile_under_dispatch'], tariff, export_tariff)
print "Total annual bill under optimal dispatch: $", annual_bill_dispatch
