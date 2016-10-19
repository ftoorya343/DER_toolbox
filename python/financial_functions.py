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
#    Vectorized.
#    
#    Inputs:
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
#    Things that would be nice to add:
#    -Sales tax basis and rate
#    -note that sales tax goes into depreciable basis
#    -Propery taxes (res can deduct from income taxes, I think)
#    -insurance
#    -add pre-tax cash flow
#    -add residential mortgage option
#    -add carbon tax revenue

#    To Do:
#    -More exhaustive checking. I have confirmed basic formulations against SAM, but there are many permutations that haven't been checked.
#    -make incentives reduce depreciable basis
#    -add a flag for high incentive levels
#    -battery price schedule, for replacements
#    -improve inverter replacement
#    -improve battery replacement
#    -add inflation adjustment for replacement prices
#    -improve deprec schedule handling
#    -Make financing unique to each agent
#    -improve the .reshape(n_agents,1) implementation
#    -Make battery replacements depreciation an input, with default of 7 year MACRS
#    -Have a better way to deal with capacity vs effective capacity and battery costs
#    '''

########################## Test Inputs ########################################
#bill_savings = np.zeros([1,26], float)
#bill_savings[:,1:] = 32007.0
#pv_size = np.array([200.0])
#pv_price = np.array([2500.0])
#inverter_price = np.array([100.0])
#pv_om = np.array([20.0])
#batt_cap = np.array([0])
#batt_power = np.array([0])
#batt_power_price = np.array([750.0])
#batt_cap_price = np.array([500.0])
#batt_chg_frac = np.array([1.0])
#batt_replacement_sch = np.array([10,15])
#batt_om = np.array([10.0])
#sector = 'com'
#itc = np.array([0.3])
#deprec_sched = np.array([0.2, .32, .192, .1152, 0.1152, .0576])
#fed_tax_rate = np.array([0.35])
#state_tax_rate = np.array([0.07])
#real_d = np.array([0.1])
#debt_fraction = np.array([0.8])
#analysis_years = 25
#inflation = 0.025
#loan_rate = np.array([0.05])
#loan_term = np.array([20])
#cash_incentives = np.array([0])
#ibi = np.array([0])
#cbi = np.array([0])
#pbi = np.array([0])

analysis_years = 25

e_escalation = np.zeros(analysis_years+1)
e_escalation[0] = 1.0
e_escalation[1:] = (1.0039)**np.arange(analysis_years)

bill_savings = np.zeros([2,26], float)
bill_savings[0,1:] = 416269.0
bill_savings[1,1:] = 416269.0
bill_savings = bill_savings*e_escalation
pv_size = np.array([2088.63, 2088.63])
pv_price = np.array([2160.0, 2160])
inverter_price = np.array([0.0, 0])
pv_om = np.array([20.0, 20])
batt_cap = np.array([521.727, 521.727])
batt_power = np.array([181.938, 181.938])
batt_power_price = np.array([1600.0, 1600.0])
batt_cap_price = np.array([500.0, 500.0])
batt_chg_frac = np.array([1.0, 1.0])
batt_replacement_sch = np.array([10,20])
batt_om = np.array([0.0, 0.0])
sector = np.array(['com', 'com'])
itc = np.array([0.3, 0.3])
deprec_sched_single = np.array([0.6, .16, .096, 0.0576, 0.0576, .0288])
deprec_sched = np.zeros([2,len(deprec_sched_single)]) + deprec_sched_single
deprec_sched_single = np.array([0.6, .16, .096, 0.0576, 0.0576, .0288])
macrs_7_yr_sch = np.array([.1429,.2449,.1749,.1249,.0893,.0892,.0893,0.0446])
fed_tax_rate = np.array([0.35, 0.35])
state_tax_rate = np.array([0.0, 0.0])
real_d = np.array([0.08, 0.08])
debt_fraction = np.array([0.0, 0.0])
inflation = 0.02
loan_rate = np.array([0.05, 0.05])
loan_term = np.array(20)
cash_incentives = np.array([0,0])
ibi = np.array([0,0])
cbi = np.array([0,0])
pbi = np.array([0,0])

    
#################### Setup #########################################
shape = (np.shape(bill_savings)[0], analysis_years+1)
nom_d = (1 + real_d) * (1 + inflation) - 1
effective_tax_rate = fed_tax_rate * (1 - state_tax_rate) + state_tax_rate
cf = np.zeros(shape) 
#inflation_adjustment = np.zeros(analysis_years+1)
#inflation_adjustment[0] = 1.0
inflation_adjustment = (1+inflation)**np.arange(analysis_years+1)
n_agents = shape[0]

#################### Bill Savings #########################################
# For C&I customers, bill savings are reduced by the effective tax rate,
# assuming the cost of electricity could have otherwise been counted as an
# O&M expense to reduce federal and state taxable income.
bill_savings = bill_savings*inflation_adjustment # Adjust for inflation
after_tax_bill_savings = np.zeros(shape)
after_tax_bill_savings = bill_savings * (1 - (sector!='res').reshape(n_agents,1)*effective_tax_rate.reshape(n_agents,1)) # reduce value of savings because they could have otherwise be written off as operating expenses

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
# It would be better to inflate the replacement costs for inflation, rather
# than adjusting it at the end.
inv_replacement_cf = np.zeros(shape)
batt_replacement_cf = np.zeros(shape)

# Inverter replacements
inv_replacement_cf[:,10] -= pv_size * inverter_price # assume a single inverter replacement at year 10

# Battery replacements
# Assumes battery replacements can harness 7 year MACRS depreciation
batt_power_price_replace = 200.0
batt_cap_price_replace = 200.0
replacement_deductions = np.zeros([n_agents,analysis_years+20]) #need a temporary larger array to hold depreciation schedules. Not that schedules may get truncated by analysis years. 
for yr in batt_replacement_sch:
    batt_replacement_cf[:,yr] -= batt_power*batt_power_price_replace + batt_cap*batt_cap_price_replace
    replacement_deductions[:,yr+1:yr+9] = batt_cost.reshape(n_agents,1) * macrs_7_yr_sch #this assumes no itc or basis-reducing incentives for batt replacements

# Adjust for inflation
inv_replacement_cf = inv_replacement_cf*inflation_adjustment
batt_replacement_cf = batt_replacement_cf*inflation_adjustment
deprec_deductions = replacement_deductions[:,:analysis_years+1]*inflation_adjustment

cf += inv_replacement_cf + batt_replacement_cf

#################### Operating Expenses ###################################
# Nominally includes O&M, fuel, insurance, and property tax - although 
# currently only includes O&M.
# All operating expenses increase with inflation
operating_expenses_cf = np.zeros(shape)
operating_expenses_cf[:,1:] = (pv_om * pv_size + batt_om * batt_cap).reshape(n_agents,1)
operating_expenses_cf = operating_expenses_cf*inflation_adjustment
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
deprec_deductions[:,1:np.size(deprec_sched,1)+1] = deprec_basis.reshape(n_agents,1) * deprec_sched
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

debt_balance[:,:loan_term] = initial_debt.reshape(n_agents,1)*((1+loan_rate).reshape(n_agents,1)**np.arange(loan_term)) - (annual_principal_and_interest_payment.reshape(n_agents,1)*(((1+loan_rate).reshape(n_agents,1)**np.arange(loan_term) - 1.0)/loan_rate.reshape(n_agents,1)))  
interest_payments[:,1:] = debt_balance[:,:-1] * loan_rate.reshape(n_agents,1)
principal_and_interest_payments[:,1:loan_term+1] = annual_principal_and_interest_payment.reshape(n_agents,1)

cf -= principal_and_interest_payments

    
#################### State Income Tax #########################################
# Per SAM, taxable income is CBIs and PBIs (but not IBIs)
# Assumes no state depreciation
# Assumes that revenue from DG is not taxable income
total_taxable_income = np.zeros(shape)
total_taxable_income[:,1] = cbi
total_taxable_income += pbi.reshape(n_agents,1)

state_deductions = np.zeros(shape)
state_deductions += interest_payments * (sector!='res').reshape(n_agents,1)
state_deductions += operating_expenses_cf

total_taxable_state_income_less_deductions = total_taxable_income - state_deductions
state_income_taxes = total_taxable_state_income_less_deductions * state_tax_rate.reshape(n_agents,1)

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
fed_income_taxes = total_taxable_fed_income_less_deductions * fed_tax_rate.reshape(n_agents,1)

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