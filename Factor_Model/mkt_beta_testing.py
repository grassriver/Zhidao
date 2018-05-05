# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:29:08 2018

@author: Hillary
"""

stk=model_data[model_data['code']=='000001']
stk=stk[['code','date','log_ret']]

mkt_ret = mkt_ret.rename(columns={'index': 'mkt_ret'})

test = pd.merge(stk,mkt_ret,on='date')

test=test[['code','date','log_ret','mkt_ret']]


def get_ols_beta(asset, market, riskfree=0.0):
    """
    Calculates r-squared ratio.

    Parameters
    ----------
    asset:
        np.series; portfolio returns.
    market: 
        np.series; market returns
    riskfree: 
        default 0.0; risk free rate, can be any floating number

    Returns
    -------
    R squared: float
        The R-squared of fitted CAPM model.
    """
    
    keep = ~(np.isnan(market) | np.isnan(asset))
    market = market[keep]
    asset = asset[keep]
    if len(market) <= 1:
        return np.nan
    if len(asset) <= 1:
        return np.nan
    y = market - riskfree
    x = asset - riskfree
    x = sm.add_constant(x)
    fit = sm.OLS(y, x).fit()
    # ddof is applied to get unbiased estimator of std deviation
    return round(fit.params, 4)


def get_beta(asset, market, riskfree=0.0):
    """
    Calculates Portfolio Beta.

    Parameters
    ----------
    asset:
        np.series; portfolio returns
    market: 
        np.series; market returns
    riskfree:
        default 0.0; risk free rate, can be any floating number

    Returns
    -------
    beta : float
        The beta value.
    """

    keep = ~(np.isnan(market) | np.isnan(asset))
    market = market[keep]
    asset = asset[keep]
    if len(market) <= 1:
        return np.nan
    if len(asset) <= 1:
        return np.nan
    cov = np.cov(market - riskfree, asset - riskfree)
    return round(cov[0, ][1] / cov[0, ][0], 4)

ols_beta = get_ols_beta(test.mkt_ret,test.log_ret,  riskfree=0.0)
cov_beta = get_beta(test.log_ret, test.mkt_ret, riskfree=0.0)