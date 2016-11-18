# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:51:27 2016

@author: pgagnon
"""

import numpy as np

def tiered_calc_vec(values, levels, prices):
    # Vectorized piecewise function calculator
    # 'Values' is a vector of the input variables to be evaluated by the piecewise function
    # 'levels' are the cumulative limits of a given tier
    # 'prices' are the cost in $/kWh for the given tier bounded by corresponding 'levels'
    # For 'levels' and 'prices', rows are tiers, columns are separate instances (periods)
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