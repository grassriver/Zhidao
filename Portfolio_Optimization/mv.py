# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 17:28:20 2018

@author: Kai Zheng
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci

#%%
def statistics(weights, mu, sigma):
    ''' 
    Returns portfolio statistics.
    
    Parameters
    ==========
    rets: pd.DataFrame
        daily returns of each stock
    weights : array-like
        weights for different securities in portfolio
    
    Returns
    =======
    pret : float
        expected portfolio return
    pvol : float
        expected portfolio volatility
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    
    weights = np.array(weights)
    pret = np.sum(mu * weights)
    pvol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
    return np.array([pret, pvol, pret / pvol])


def min_func_sharpe(weights, mu, sigma):
    '''
    Return negative value of the Sharpe ratio.
    Parameters
    ==========
    
    '''
    return -statistics(weights, mu, sigma)[2]

def min_func_variance(weights, mu, sigma):
    '''
    expected variance
    '''
    return statistics(weights, mu, sigma)[1] ** 2

#constraint the expected return to a target return
def min_func_port(weights, mu, sigma):
    '''
    expected volatility
    '''
    return statistics(weights, mu, sigma)[1]

def f(x, tck):
    ''' 
    Efficient frontier function (splines approximation). 
    '''
    return sci.splev(x, tck, der=0)

def df(x, tck):
    '''
    First derivative of efficient frontier function.
    '''
    return sci.splev(x, tck, der=1)

def equations(p, tck, rf=0.01):
    '''
    equation for the market portfolio and riskfree asset
    
    Parameters
    ==========
    p=[a,b,t]
        expected return = a + bx
        t is corrsponding x for tangent portfolio
    '''
    eq1 = rf - p[0]
    eq2 = p[0] + p[1]*p[2] - f(p[2], tck)
    eq3 = p[1] - df(p[2], tck)
    return eq1, eq2, eq3

def simulation(mu, sigma):
    '''
    Monte-Carlo Simulation: 
        simulate portfolio mean and variance with randomly generated weights
    '''
    noa = len(mu) # number of assets
    prets = []
    pvols = []
    for p in range(2500):
        weights = np.random.random(noa)
        weights = weights/np.sum(weights)
        prets.append(np.sum(mu*weights))
        pvols.append(np.sqrt(np.dot(weights.T,
                                    np.dot(sigma, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)
    return prets, pvols

def opt_s_v(mu, sigma):
    '''
    maximize sharpe ration and minimize volatility
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #constraints
    bnds = tuple((0,1) for x in range(noa))#bounds for parameters
    init = noa*[1. / noa,] # initial weights: multiplication of a list returns replications of this list
    # maximize sharpe ratio port
    opts = sco.minimize(min_func_sharpe, init, args=(mu,sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    # global minimum variance port
    optv = sco.minimize(min_func_variance, init, args=(mu,sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opts, optv

def eff_fter(mu, sigma):
    '''
    effcient frontier
    '''
    noa = len(mu)
    bnds = tuple((0,1) for x in range(noa)) #bounds for parameters
    init = noa*[1. / noa,]
    trets = np.linspace(np.min(mu), np.max(mu), 50)
    tvols = []
    for tret in trets:
        cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[0] - tret},
                 {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(min_func_port, init, args=(mu, sigma), method='SLSQP',
                           bounds=bnds, constraints=cons)
        if ((np.abs(statistics(res.x, mu, sigma)[0] - tret)<1e-2) &
            (np.abs(np.sum(res.x) - 1)<1e-3)):
            tvols.append(res['fun'])
        else:
            tvols.append(np.nan)
    tvols = np.array(tvols)
    trets = trets[~np.isnan(tvols)] # drop na
    tvols = tvols[~np.isnan(tvols)] # drop na
    return trets, tvols

def cml(mu, sigma):
    '''
    Solve the capital market line
    '''
    # get the efficient frontier
    trets, tvols = eff_fter(mu, sigma)
    
    # solove the capital market line
    ind = np.argmin(tvols)
    evols = tvols[ind:]
    erets = trets[ind:]
    tck = sci.splrep(evols, erets, k=2) # interporate the efficinet frontier
    opt = sco.fsolve(equations, [0.01, 0.7, 0.8], args=(tck,0.01))    
    return opt, tck, evols, erets

def tan_port(mu, sigma):
    '''
    weights of tangent portfolio
    
    Returns
    =======
    optt: OptimizeResult
        portfolio weights
    '''
    opt, tck , _, _ = cml(mu, sigma)
    noa = len(mu)
    eret = f(opt[2], tck)
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[0] - eret},
             {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #constraints
    bnds = tuple((0,1) for x in range(noa))#bounds for parameters
    init = noa*[1. / noa,] # initial weights: multiplication of a list returns replications of this list
    
    # maximize sharpe ratio port
    optt = sco.minimize(min_func_port, init, args=(mu,sigma), method='SLSQP', 
                        bounds=bnds, constraints=cons)
    return optt
    
def plot_eff_fter(mu, sigma):
    '''
    Plot the efficient frontier and capital market line.
    '''
    opt, tck, evols, erets = cml(mu, sigma)
    prets, pvols = simulation(mu, sigma)
    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets,
                c=(prets - 0.01) / pvols, marker='o')
    # random portfolio composition
    plt.plot(evols, erets, 'g', lw=4.0)
    # efficient frontier
    cx = np.linspace(0.0, 0.5)
    plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
    # capital market line
    plt.plot(opt[2], f(opt[2], tck), 'r*', markersize=15.0)
    plt.grid(True)
    plt.axhline(0, color='k', ls='--', lw=2.0)
    plt.axvline(0, color='k', ls='--', lw=2.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')