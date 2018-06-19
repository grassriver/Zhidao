# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 17:28:20 2018

@author: Kai Zheng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
from stock_class import Stock
import warnings

#%%


def statistics(weights, mu, sigma, rf=0.03):
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
        Sharpe ratio for rf=0.01
    '''

    weights = np.array(weights)
    mu = np.array(mu)
    sigma = np.array(sigma)
    pret = np.sum(mu * weights)
    pvol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
    return np.array([pret, pvol, (pret - rf) / pvol])


def min_func_sharpe(weights, mu, sigma, rf=0.03):
    '''
    Return negative value of the Sharpe ratio.
    Parameters
    ==========

    '''
    return -statistics(weights, mu, sigma, rf)[2]


def min_func_variance(weights, mu, sigma):
    '''
    expected variance
    '''
    return statistics(weights, mu, sigma)[1] ** 2

# constraint the expected return to a target return


def min_func_vol(weights, mu, sigma):
    '''
    expected volatility
    '''
    return statistics(weights, mu, sigma)[1]

def max_func_ret(weights, mu, sigma):
    '''
    return negative value of the expected return
    '''
    return -statistics(weights, mu, sigma)[0]

def max_quadratic(weights, mu, sigma, l, rf):
    '''
    return negative value of quadratic utility
    '''
    pret, pvol, _ = statistics(weights, mu, sigma)
    return  - pret + l*(pvol**2) - (1-weights.sum())*rf

def f(x, tck):
    ''' 
    Efficient frontier function (splines approximation). 
    '''
    return sci.splev(x, tck, der=0)


def df(x, tck):
    '''
    First derivative of efficient frontier function (splines approximation).
    '''
    return sci.splev(x, tck, der=1)


def equations(p, tck, rf=0.03):
    '''
    equation for the market portfolio and riskfree asset

    Parameters
    ==========
    p=[a,b,t]
        expected return = a + bx
        t is corrsponding x for tangent portfolio
    '''
    eq1 = rf - p[0]
    eq2 = p[0] + p[1] * p[2] - f(p[2], tck)
    eq3 = p[1] - df(p[2], tck)
    return eq1, eq2, eq3


def simulation(mu, sigma):
    '''
    Monte-Carlo Simulation: 
        simulate portfolio mean and variance with randomly generated weights
    '''
    noa = len(mu)  # number of assets
    mu = np.array(mu)
    sigma = np.array(sigma)
    prets = []
    pvols = []
    for p in range(2500):
        weights = np.random.random(noa)
        weights = weights / np.sum(weights)
        prets.append(np.sum(mu * weights))
        pvols.append(np.sqrt(np.dot(weights.T,
                                    np.dot(sigma, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)
    return prets, pvols


def opt_s_v(mu, sigma, rf=0.03):
    '''
    maximize sharpe ratio and minimize volatility
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list
    # maximize sharpe ratio port
    opts = sco.minimize(min_func_sharpe, init, args=(mu, sigma, rf), method='SLSQP',
                        bounds=bnds, constraints=cons)
    # global minimum variance port
    optv = sco.minimize(min_func_variance, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opts, optv

def opt_s(mu, sigma, rf=0.03):
    '''
    maximize sharpe ratio
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list
    # maximize sharpe ratio port
    opts = sco.minimize(min_func_sharpe, init, args=(mu, sigma, rf), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opts.x

def opt_v(mu, sigma, rf=0.03):
    '''
    minimize volatility
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list
    # global minimum variance port
    optv = sco.minimize(min_func_variance, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return optv.x

def eff_fter(mu, sigma):
    '''
    effcient frontier
    '''
    noa = len(mu)
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]
    trets = np.linspace(np.min(mu), np.max(mu), 100)
    tvols = []
    for tret in trets:
        cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[0] - tret},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(min_func_vol, init, args=(mu, sigma), method='SLSQP',
                           bounds=bnds, constraints=cons)
        if ((np.abs(statistics(res.x, mu, sigma)[0] - tret) < 1e-2) &
                (np.abs(np.sum(res.x) - 1) < 1e-3)):
            tvols.append(res['fun'])
        else:
            tvols.append(np.nan)
    tvols = np.array(tvols)
    trets = trets[~np.isnan(tvols)]  # drop na
    tvols = tvols[~np.isnan(tvols)]  # drop na
    return trets, tvols


def cml_eff(mu, sigma, rf=0.03):
    '''
    Solve the capital market line and interpolate the efficient frontier
    '''
    # get the efficient frontier
    trets, tvols = eff_fter(mu, sigma)

    # solove the capital market line
    ind = np.argmin(tvols)
    evols = tvols[ind:]
    erets = trets[ind:]
    tck = sci.splrep(evols, erets, k=2)  # interporate the efficinet frontier
    opt = sco.fsolve(equations, [0.01, 0.7, 0.8], args=(tck, rf))
    return opt, tck, evols, erets


def tan_port(mu, sigma, rf=0.03):
    '''
    weights of tangent portfolio (according to the intersection point
    of efficient frontier and cml)

    Returns
    =======
    optt: OptimizeResult
        portfolio weights
    
    Note that theoretically speaking, this portfolio should be identical to 
    the portfolio which maximize the sharpe ratio. But due to the computational error of 
    interpolation, they could be quite different. 
    '''
    opt, tck, _, _ = cml_eff(mu, sigma)
    noa = len(mu)
    eret = f(opt[2], tck) # expected return for the tangent point
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[0] - eret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    optt = sco.minimize(min_func_vol, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return optt


def plot_eff_fter(mu, sigma, rf=0.03):
    '''
    Plot the efficient frontier and capital market line.
    '''
    opt, tck, evols, erets = cml_eff(mu, sigma)
    prets, pvols = simulation(mu, sigma)
    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets,
                c=(prets - 0.01) / pvols, marker='o')
    # random portfolio composition
    plt.plot(evols, erets, 'g', lw=4.0) # efficient frontier

    
    opts, optv = opt_s_v(mu, sigma, rf)
    weights = opts.x
    pret, pvol, _ = statistics(weights, mu, sigma, rf)
    plt.plot(pvol, pret, 'b*', markersize=15.0)
    
#    tp_weights = tan_port(mu, sigma).x
#    pret, pvol, _ = statistics(tp_weights, mu, sigma, rf)
#    plt.plot(pvol, pret, 'y*', markersize=15.0)
    
#    max_vol = np.max(np.sqrt(np.diagonal(sigma)))
#    cx = np.linspace(0.0, max_vol)
#    plt.plot(cx, opt[0] + (pret-opt[0])/pvol*cx, lw=1.5) # capital market line
    
#    plt.plot(opt[2], f(opt[2], tck), 'r*', markersize=15.0)
    plt.grid(True)
    plt.axhline(0, color='k', ls='--', lw=2.0)
    plt.axvline(0, color='k', ls='--', lw=2.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')


def hist_expect(conn, code_list, start='2016-01-01', end='2017-01-01', 
                backfill=False, stocks_price_old=None, business_calendar=None,
                industry=None):
    '''
    calculate expected return and covariance matrix from historical return.

    Returns
    =======
    emu: pandas.Series
        expected return
    esigma: pandas.DataFrame
        expected covariance matrix
    '''
    s = Stock(conn, code_list, start, end, backfill=backfill, 
              stocks_price_old=stocks_price_old,
              business_calendar=business_calendar,
              industry=industry)
    rets = s.daily_returns
    emu = rets.mean() * 252
    esigma = rets.cov() * 252
    return emu, esigma

def max_sharpe(mu, sigma, rf=0.03):
    '''
    weights of portfolio that maximize sharpe ratio
    '''
    opts, _ = opt_s_v(mu, sigma, rf)
    return opts.x

def min_vol(mu, sigma, rf=0.03):
    '''
    weights of GMV
    '''
    _, optv = opt_s_v(mu, sigma, rf)
    return optv.x

def max_ret_st_vol(tvol, mu, sigma):
    '''
    maximize expected return subject to targeted volatility without risk free asset
    '''
    _, optv = opt_s_v(mu, sigma)
    min_vol = statistics(optv.x, mu, sigma)[1]
    max_vol = np.sqrt(np.diag(sigma)).max().max()
    if tvol<min_vol:
        tvol = min_vol
        warnings.warn('tvol set to be min_vol, tvol should be between {0} and {1}'.format(min_vol, max_vol))
    elif tvol>max_vol:
        tvol = max_vol
        warnings.warn('tvol set to be max_vol, tvol should be between {0} and {1}'.format(min_vol, max_vol))
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[1] - tvol},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(max_func_ret, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opt.x

    
def min_vol_st_ret(tret, mu, sigma):
    '''
    minimize expected volatility subject to targeted return without risk free asset
    '''
    if tret>np.max(mu) or tret<np.min(mu):
        raise ValueError('tret should be between {0} and {1}'.format(np.min(mu), np.max(mu)))
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[0] - tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(min_func_vol, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opt.x

def max_ret_st_vol_wrf(tvol, mu, sigma, rf=0.03):
    '''
    maximize expected return subject to targeted volatility with risk free asset
    '''
    opts, _ = opt_s_v(mu, sigma, rf)
    if tvol <= 0 or tvol > np.max(np.sqrt(sigma)):
        raise ValueError('tvol should be between {0} and {1}'.format(0, np.max(np.sqrt(sigma))))
    w_M = opts.x
    mvol = statistics(w_M, mu, sigma)[1]
    w = w_M * tvol / mvol
    return w

def min_vol_st_ret_wrf(tret, mu, sigma, rf=0.03):
    '''
    minimize expected volatility subject to targeted return with risk free asset
    '''
    if tret>np.max(mu) or tret<rf:
        raise ValueError('tret should be between {0} and {1}'.format(rf, np.max(mu)))
    opts, _ = opt_s_v(mu, sigma, rf)
    w_M = opts.x
    mret = statistics(w_M, mu, sigma)[0]
    w = w_M * (tret - rf) / (mret - rf)
    return w


def max_ret_st_vol_restricted(tvol, mu, sigma, w0):
    '''
    maximize expected return subject to targeted volatility without risk free asset
    restrictions:
        1. number of stocks is no more than 10
        2. the largest difference between new weight and old weight is no more than 20%
    '''
    _, optv = opt_s_v(mu, sigma)
    min_vol = statistics(optv.x, mu, sigma)[1]
    if tvol<min_vol or tvol>np.sqrt(np.diag(sigma)).max().max():
        raise ValueError('tvol should be between {0} and {1}'.format(min_vol, np.sqrt(sigma).max().max()))
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[1] - tvol},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: 0.2 - np.max(np.abs(x-w0))})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(max_func_ret, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    w = opt.x
    # retain top ten stocks in weights
    if noa > 10:
        w[np.argsort(w)[:-10]]=0
        w = w/np.sum(w)
    return w

def min_vol_st_ret_resticted(tret, mu, sigma, w0):
    '''
    minimize expected volatility subject to targeted return without risk free asset
    '''
    if tret>np.max(mu) or tret<np.min(mu):
        raise ValueError('tret should be between {0} and {1}'.format(np.min(mu), np.max(mu)))
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x, mu, sigma)[0] - tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: 0.2 - np.max(np.abs(x-w0))})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(min_func_vol, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    w = opt.x
    # retain top ten stocks in weights
    if noa > 10:
        w[np.argsort(w)[:-10]]=0
        w = w/np.sum(w)
    return w

def opt_s_restricted(mu, sigma, w0, rf=0.03, max_turnover=0.2):
    '''
    maximize sharpe ration and minimize volatility
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: max_turnover - np.max(np.abs(x-w0))})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list
    # maximize sharpe ratio port
    opts = sco.minimize(min_func_sharpe, init, args=(mu, sigma, rf), method='SLSQP',
                        bounds=bnds, constraints=cons)
    w = opts.x
    # retain top ten stocks in weights
    if noa > 10:
        w[np.argsort(w)[:-10]]=0
        w = w/np.sum(w)
    return w

def opt_v_restricted(mu, sigma, w0, rf=0.03, max_turnover=0.2):
    '''
    maximize sharpe ration and minimize volatility
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: max_turnover - np.max(np.abs(x-w0))})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list
    # global minimum variance port
    optv = sco.minimize(min_func_variance, init, args=(mu, sigma), method='SLSQP',
                        bounds=bnds, constraints=cons)
    w = optv.x
    # retain top ten stocks in weights
    if noa > 10:
        w[np.argsort(w)[:-10]]=0
        w = w/np.sum(w)
    return w

def opt_quadratic(mu, sigma, l):
    '''
    maximize expected return subject to targeted volatility without risk free asset
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(max_quadratic, init, args=(mu, sigma, l, 0), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opt.x

def opt_quadratic_risky(mu, sigma, l, rf=0.03):
    '''
    maximize expected return subject to targeted volatility without risk free asset
    '''
    noa = len(mu)
    cons = ({'type': 'ineq', 'fun': lambda x: - np.sum(x) + 1})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(max_quadratic, init, args=(mu, sigma, l, rf), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opt.x

def opt_quadratic_risky1(mu, sigma, l, rf=0.03):
    '''
    maximize expected return subject to targeted volatility without risk free asset
    '''
    noa = len(mu)
    cons = ({'type': 'ineq', 'fun': lambda x: - np.sum(x) + 1})  # constraints
    bnds = tuple((0, 0.1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(max_quadratic, init, args=(mu, sigma, l, rf), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opt.x

def opt_quadratic_risky2(mu, sigma, l, rf=0.03):
    '''
    maximize expected return subject to targeted volatility without risk free asset
    '''
    noa = len(mu)
    cons = ({'type': 'eq', 'fun': lambda x: - np.sum(x) + 1})  # constraints
    bnds = tuple((0, 0.1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(max_quadratic, init, args=(mu, sigma, l, rf), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opt.x

def opt_quadratic_risky_restricted(mu, sigma, l, w0=None, rf=0.03, max_turnover=0.2):
    '''
    maximize expected return subject to targeted volatility without risk free asset
    '''
    noa = len(mu)
    if w0 is None:
        cons = ({'type': 'ineq', 'fun': lambda x: - np.sum(x) + 1})  # constraints
    else:    
        cons = ({'type': 'ineq', 'fun': lambda x: - np.sum(x) + 1},
                {'type': 'ineq', 'fun': lambda x: max_turnover - np.max(np.abs(x-w0))})  # constraints
    bnds = tuple((0, 1) for x in range(noa))  # bounds for parameters
    init = noa * [1. / noa, ]  # initial weights: multiplication of a list returns replications of this list

    # get the corresponding portfolio on the efficient frontier
    opt = sco.minimize(max_quadratic, init, args=(mu, sigma, l, rf), method='SLSQP',
                        bounds=bnds, constraints=cons)
    return opt.x

def hist_expect_mu(conn, code_list, start='2016-01-01', end='2017-01-01', 
                   backfill=False, stocks_price_old=None, business_calendar=None,
                   industry=None, **kwargs):
    '''
    calculate annualized expected return from historical return.

    Returns
    =======
    emu: pandas.Series
        expected return
    '''
    s = Stock(conn, code_list, start, end, backfill=backfill, 
              stocks_price_old=stocks_price_old,
              business_calendar=business_calendar,
              industry=industry)
    rets = s.daily_returns
    emu = rets.mean() * 252
    return emu

def hist_expect_sigma(conn, code_list, start='2016-01-01', end='2017-01-01', 
                      backfill=False, stocks_price_old=None, business_calendar=None,
                      industry=None, **kwargs):
    '''
    calculate annualized covariance matrix from historical return.

    Returns
    =======
    esigma: pandas.DataFrame
        expected covariance matrix
    '''
    s = Stock(conn, code_list, start, end, backfill=False, 
              stocks_price_old=stocks_price_old,
              business_calendar=business_calendar,
              industry=industry)
    rets = s.daily_returns

    esigma = rets.cov() * 252
    
    # if not enough close price data, covariance should not be calculated
    temp = (~pd.isnull(rets)).sum()
    idx = (temp[temp<10].index).tolist()
    if idx:
        print(start)
        print(end)
        raise ValueError('Data less than 10 days for stocks {}!'.format(','.join(idx)))
    # deal with 0 correlations
    #    var = np.diag(esigma)
    
    return esigma

#def residual_alpha(conn, code_list, start='2016-01-01', end='2017-01-01', 
#                   backfill=False, stocks_price_old=None, business_calendar=None):
#    rets = hist_expect_mu(conn, code_list, start, end, 
#                          backfill=backfill, 
#                          stocks_price_old=stocks_price_old, 
#                          business_calendar=business_calendar, 
#                          industry=industry)
#    beta = get beta
#    residual = rets - beta*rm
#    return residual

