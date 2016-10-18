# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:27:48 2016

@author: pgagnon
"""

import numpy as np

#def cashflow_constructor(bill_savings, 
#                         pv_size, pv_price, inverter_price, pv_om,
#                         batt, batt_power_price, batt_cap_price, batt_chg_frac,
#                         batt_replacement_sch, batt_om,
#                         sector, itc, deprec_sched, 
#                         fed_tax_rate, state_tax_rate, real_d, debt_fraction, 
#                         analysis_years, inflation, 
#                         loan_rate, loan_term, 
#                         cash_incentives=0, ibi=0, cbi=0, pbi=0):
#    '''
#    Accepts financial assumptions and returns the cash flows for the projects.
#    Vectorized (not yet, actually).
#    
#    Inputs:
#    -fa is financial assumptions class
#    -bill_savings is a cash flow of the annual bill savings over the lifetime
#     of the system, including changes to the price or structure of electricity, 
#     in present-year dollars (i.e., excluding inflation).
#     Need to construct this beforehand, to either simulate degradation or make 
#     assumptions about degradation.
#    -ibi is up front investment based incentive
#    -cbi is up front capacity based incentive
#    -batt_chg_frac is the fraction of the battery's energy that it gets from
#     a co-hosted PV system. Used for ITC calculation.
#    
#    To Do:
#    -check the signs on all the math
#    -vectorize
#    -Sales tax basis and rate
#    -note that sales tax goes into depreciable basis
#    -Grid interconnection and other fixed fees
#    -Propery taxes (res can deduct from income taxes, I think)
#    -generalized incentives
#    -payback period, LCOE
#    -insurance
#    -make incentives reduce depreciable basis
#    -add a flag for high incentive levels
#    -add rebates and other incentive forms?
#    -have batts replace on boolean schedule
#    -battery price schedule, for replacements
#    -improve inverter replacement
#    -add inflation
#    -add pre-tax cash flow
#    '''

########################## Test Inputs ########################################
bill_savings = np.zeros([1,26], float)
bill_savings[:,1:] = 32007.0
pv_size = np.array([200.0])
pv_price = np.array([2500.0])
inverter_price = np.array([100.0])
pv_om = np.array([20.0])
batt_cap = np.array([0])
batt_power = np.array([0])
batt_power_price = np.array([750.0])
batt_cap_price = np.array([500.0])
batt_chg_frac = np.array([1.0])
batt_replacement_sch = np.array([10,15])
batt_om = np.array([10.0])
sector = 'com'
itc = np.array([0.3])
deprec_sched = np.array([0.2, .32, .192, .1152, 0.1152, .0576])
fed_tax_rate = np.array([0.35])
state_tax_rate = np.array([0.07])
real_d = np.array([0.1])
debt_fraction = np.array([0.8])
analysis_years = 25
inflation = 0.025
loan_rate = np.array([0.05])
loan_term = np.array([20])
cash_incentives = np.array([0])
ibi = np.array([0])
cbi = np.array([0])
pbi = np.array([0])

    
#################### Setup #########################################
shape = (np.shape(bill_savings)[0], analysis_years+1)
nom_d = (1 + real_d) * (1 + inflation) - 1
effective_tax_rate = fed_tax_rate * (1 - state_tax_rate) + state_tax_rate
cf = np.zeros(shape) 

#################### Bill Savings #########################################
# For C&I customers, bill savings are reduced by the effective tax rate,
# assuming the cost of electricity could have otherwise been counted as an
# O&M expense to reduce federal and state taxable income.
after_tax_bill_savings = np.zeros(shape)
after_tax_bill_savings = bill_savings * (1 - (sector!='res')*effective_tax_rate) # reduce value of savings because they could have otherwise be written off as operating expenses
cf += after_tax_bill_savings

#################### Installed Costs ######################################
# Assumes that cash incentives, IBIs, and CBIs will be monetized in year 0,
# reducing the up front installed cost that determines debt levels. 
pv_cost = pv_size*pv_price     # assume pv_price includes initial inverter purchase
batt_cost = batt_power*batt_power_price + batt_cap*batt_cap_price
installed_cost = pv_cost + batt_cost
net_installed_cost = installed_cost - cash_incentives - ibi - cbi
up_front_cost = net_installed_cost * (1 - debt_fraction)
cf[:,0] -= up_front_cost

#################### Replacements #########################################
inv_replacement_cf = np.zeros(shape)
batt_replacement_cf = np.zeros(shape)
inv_replacement_cf[:,10] -= pv_size * inverter_price # assume a single inverter replacement at year 10
deprec_deductions = np.zeros(shape)
for yr in batt_replacement_sch:
    batt_replacement_cf[:,yr] -= batt_cost
    deprec_deductions[:,yr+1:yr+1+len(deprec_sched)] = batt_cost * deprec_sched #this assumes no itc or basis-reducing incentives for batt replacements
cf += inv_replacement_cf + batt_replacement_cf

#################### Operating Expenses ###################################
# Includes O&M, fuel, insurance, property tax
operating_expenses_cf = np.zeros(shape)
operating_expenses_cf[:,1:] = pv_om * pv_size + batt_om * batt_cap
cf -= operating_expenses_cf

#################### Federal ITC #########################################
pv_itc_value = pv_cost * itc
batt_itc_value = batt_cost * itc * batt_chg_frac * (batt_chg_frac>=0.75)
itc_value = pv_itc_value + batt_itc_value
# itc value added in fed_tax_savings_or_liability

#################### Depreciation #########################################
# Per SAM, depreciable basis is sum of total installed cost and total 
# construction financing costs, less 50% of ITC and any incentives that
# reduce the depreciable basis.
deprec_basis = installed_cost - itc_value*0.5 
deprec_deductions[:,1:len(deprec_sched)+1] = deprec_basis * deprec_sched
# to be used later in fed tax calcs

#################### Debt cash flow #######################################
# Deduct loan interest payments from state & federal income taxes for res 
# mortgage and C&I. No deduction for res loan.
# note that the debt balance in year0 is different from principal if there 
# are any ibi or cbi. Not included here yet.
# debt balance, interest payment, principal payment, total payment

initial_debt = net_installed_cost - up_front_cost
annual_principal_and_interest_payment = initial_debt * (loan_rate*(1+loan_rate)**loan_term) / ((1+loan_rate)**loan_term - 1)
debt_balance = np.zeros(shape)
interest_payments = np.zeros(shape)
principal_and_interest_payments = np.zeros(shape)

debt_balance[:,:loan_term] = (initial_debt*(1+loan_rate)**np.arange(loan_term)) - (annual_principal_and_interest_payment*(((1+loan_rate)**np.arange(loan_term) - 1.0)/loan_rate))  
interest_payments[:,1:] = debt_balance[:,:-1] * loan_rate
principal_and_interest_payments[:,1:loan_term+1] = annual_principal_and_interest_payment

cf -= principal_and_interest_payments

    
#################### State Income Tax #########################################
# Per SAM, taxable income is CBIs and PBIs (but not IBIs)
# Assumes no state depreciation
# Assumes that revenue from DG is not taxable income
total_taxable_income = np.zeros(shape)
total_taxable_income[:,1] = cbi
total_taxable_income += pbi

state_deductions = np.zeros(shape)
state_deductions += interest_payments * (sector!='res')
state_deductions += operating_expenses_cf

total_taxable_state_income_less_deductions = total_taxable_income - state_deductions
state_income_taxes = total_taxable_state_income_less_deductions * state_tax_rate

state_tax_savings_or_liability = -state_income_taxes

cf += state_tax_savings_or_liability
    
################## Federal Income Tax #########################################
# Assumes all deductions are federal
fed_deductions = np.zeros(shape)
fed_deductions += interest_payments
fed_deductions += deprec_deductions
fed_deductions += state_income_taxes
fed_deductions += operating_expenses_cf

total_taxable_fed_income_less_deductions = total_taxable_income - fed_deductions
fed_income_taxes = total_taxable_fed_income_less_deductions * fed_tax_rate

fed_tax_savings_or_liability = -fed_income_taxes
fed_tax_savings_or_liability[:,1] += itc_value

cf += fed_tax_savings_or_liability


########################### Package Results ###############################

results = {'cf':cf}
    
    ################### Financial Metrics #########################################
    # First element in cf is year 0, therefore it is not discounted
#    npv = sum(cf * (1/(1+nom_d)**np.array(range(analysis_years+1))))
#    WACC = real_d * (1 - debt_fraction) + (1 - effective_tax_rate) * loan_rate * debt_fraction
    
#    return results