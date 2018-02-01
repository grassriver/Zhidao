
# coding: utf-8

# In[25]:


# import empyrical as em
import statsmodels.api as sm


# In[26]:


import sqlite3
import numpy as np
import pandas
import math


# In[229]:


def get_return(df, code, start, end):
    series = df.loc[(df.loc[:, 'Date'] <= end) & (df.loc[:, 'Date'] >= start), code]
    price_series = pandas.DataFrame({'price': series,
                                     'lag_price': series.shift(1),
                                     'price_return': np.log(series) - np.log(series.shift(1))})
    price_series = price_series.dropna()
    return(price_series.loc[:, 'price_return'])


# In[230]:


def get_portfolio_return(stock, weight):
    stock = stock.fillna(0)
    return (stock * weight).apply(sum, 1)


# In[231]:


# Beta Calculation
# Asset: Stock Daily Return Series
# Market: Market Daily Return
# RiskFree: Risk Free Rate, default 0

def get_beta(asset, market, riskfree=0.0):
    keep = ~(np.isnan(market) | np.isnan(asset))
    market = market[keep]
    asset = asset[keep]
    if len(market) <= 1:
        return np.nan
    if len(asset) <= 1:
        return np.nan
    cov = np.cov(market - riskfree, asset - riskfree)
    return round(cov[0, ][1] / cov[0, ][0], 4)


# In[232]:


# alpha Calculation
# asset: Stock Daily return Series
# market: market Daily return
# RiskFree: Risk Free Rate, default 0
# annulization: annualization factor

def get_annulized_alpha(asset, market, annualization, riskfree=0.0):
    keep = ~(np.isnan(market) | np.isnan(asset))
    market = market[keep]
    asset = asset[keep]
    if len(market) <= 1:
        return np.nan
    if len(asset) <= 1:
        return np.nan
    beta = get_beta(market, asset, riskfree)
    alpha = np.mean((asset - riskfree) - beta * (market - riskfree))
    return round(alpha * annualization, 4)


# In[246]:


# Sharpe Ratio Calculation
# Asset: Stock Daily Return Series
# RiskFree: Risk Free Rate, default 0
# ANNUALIZATION: annualization factor

def get_sharpe_ratio(asset, annualization, riskfree=0.0):
    keep = ~np.isnan(asset)
    asset = asset[keep]
    if len(asset) <= 1:
        return np.nan
    # ddof is applied to get unbiased estimator of std deviation
    return round(((np.mean(asset) - riskfree) / np.std(asset, ddof=1)) * np.sqrt(annualization), 4)


# In[234]:


# R square Calculation
# Asset: Stock Daily Return Series
# RiskFree: Risk Free Rate, default 0
# Market: Market Daily Return

def get_rsquare(asset, market, riskfree=0.0):
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
    return round(fit.rsquared, 4)


# In[235]:


# Adj R square Calculation
# Asset: Stock Daily Return Series
# RiskFree: Risk Free Rate, default 0
# Market: Market Daily Return

def get_adj_rsquare(asset, market, riskfree=0.0):
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
    return round(fit.rsquared_adj, 4)


# In[236]:


# Sortino ratio Calculation
# Asset: Stock Daily Return Series
# RiskFree: Risk Free Rate, default 0
# ANNUALIZATION: annualization factor

def get_sortino_ratio(asset, annualization, riskfree=0.0):
    keep = ~np.isnan(asset)
    asset = asset[keep]
    if len(asset) <= 1:
        return np.nan
    # ddof is applied to get unbiased estimator of std deviation
    downside_ret = asset[asset < riskfree]
    if len(downside_ret) < 1:
        return np.nan
    downside_std = np.sqrt(1 / len(asset) * sum((downside_ret - riskfree)**2))
    return round(((np.mean(asset) - riskfree) / downside_std) * np.sqrt(annualization), 4)


# In[245]:


# Treynor ratio Calculation
# Asset: Stock Daily Return Series
# Market: Market Daily Return Series
# ANNUALIZATION: annualization factor

def get_treynor_ratio(asset, market, annualization, riskfree=0.0):
    keep = ~(np.isnan(market) | np.isnan(asset))
    market = market[keep]
    asset = asset[keep]
    if len(market) <= 1:
        return np.nan
    if len(asset) <= 1:
        return np.nan
    beta = get_beta(asset, market, riskfree)
    return round(((np.mean(asset) - riskfree) / beta) * np.sqrt(annualization), 4)


# In[238]:


# Correlation Calculation
# Asset: Stock Daily Return Series
# Market: Market Daily Return Series

def get_correlation(asset, market):
    keep = ~(np.isnan(market) | np.isnan(asset))
    market = market[keep]
    asset = asset[keep]
    if len(market) <= 1:
        return np.nan
    if len(asset) <= 1:
        return np.nan
    return round(np.corrcoef(asset, market)[0, 1], 4)


# In[239]:


# Diversification ratio Calculation
# Stocks: Stock Daily Return Dataframe
# Weight: Stock Weight Dataframe

def get_diversification_ratio(stock_ret, portfolio_ret, weight):
    portfolio_vol = np.std(portfolio_ret, ddof=1)
    stock_vol = stock_ret.apply(np.nanstd, ddof=1)
    weighted_stock_vol = sum(stock_vol * weight)
    return round(weighted_stock_vol / portfolio_vol, 4)

# In[240]:


# Positive Period
# Asset: Stock Daily Return Series

def positive_period(asset):
    asset = asset.dropna()
    total_count = (len(asset))
    positive_count = (np.sum(asset > 0))
    ratio = str(round((positive_count / total_count) * 100, 2))
    return (str(positive_count) + ' out of ' + str(total_count) + ' (' + ratio + '%)')


# In[241]:


# Gain Loss Ratio
# Asset: Stock Daily Return Series
def gain_loss_ratio(asset):
    asset = asset.dropna()
    gain = np.mean(asset[asset > 0])
    loss = -np.mean(asset[asset < 0])
    return round(gain / loss, 4)


def information_ratio(asset,market):
    keep = ~(np.isnan(market) | np.isnan(asset))
    market = market[keep]
    asset = asset[keep]
    if len(market) <= 1:
        return np.nan
    if len(asset) <= 1:
        return np.nan
    spread = asset - market
    info_ratio = np.nanmean(spread)/np.nanstd(spread,ddof = 1)
    return info_ratio
# In[2]:


# Sample Run
#close = pandas.read_csv('/Users/zifandeng/Nustore Files/PI/Staff Working File/X. Jin/CombClose.csv')
#ret_index = get_return(close,'sh000001','2015-01-01','2016-01-01')
# ret_price = pandas.DataFrame({"600010":get_return(close,'600010','2015-01-01','2016-01-01'),
#                             "600011":get_return(close,'600011','2015-01-01','2016-01-01')})
#stock = pandas.DataFrame.assign(ret_price)

#et_price.loc[:,'portfolio'] = get_portfolio_return(ret_price,[0.3,0.7])

# matrix = pandas.DataFrame({'Beta':ret_price.apply(get_beta,market = ret_index),
#                           'Annualized Alpha':ret_price.apply(get_annulized_alpha,market = ret_index,annualization = 252),
#                           'R Square':ret_price.apply(get_rsquare,market = ret_index),
#                           'Adj R Square':ret_price.apply(GetAdjRSquare,Market = ret_index),
#                           'Martket Correlation':ret_price.apply(get_correlation,market = ret_index),
#                           'Sharpe Ratio':ret_price.apply(get_sharpe_ratio,annualization=252),
#                           'Sortino Ratio':ret_price.apply(get_sortino_ratio,annualization=252),
#                           'Treynor Ratio':ret_price.apply(get_treynor_ratio,market=ret_index,annualization=252)})
# matrix


# In[270]:


# Sample Test 2
#stock = pandas.read_csv('/Users/zifandeng/Desktop/try1.csv')
# Diversification Ratio Sample Run
#div_ratio =get_diversification_ratio(stocks=stock,weight = [0.3,0.7])
# div_ratio


# In[273]:


# Gain Loss Ratio Sample Run
# gain_loss_ratio(stock.iloc[:,0])


# In[274]:


# Positive Perid Sample Run
# positive_period(stock.iloc[:,0])