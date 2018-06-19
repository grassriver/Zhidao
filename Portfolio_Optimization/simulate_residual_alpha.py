# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:04:22 2018

@author: Kai Zheng
"""
import numpy as np
import pandas as pd
from stock_class import Stock

def normalize(x):
    return (x-x.mean())/x.std()


def simu_res_alpha(x1, rho):
    x1 = np.matrix(x1)
    if x1.shape[0]==1:
        n = x1.shape[1]
    else:
        n = x1.shape[0]
        x1 = x1.T
    x2 = np.matrix(2+0.5*np.random.randn(n))
    x1_hat = normalize(x1)
    
    P = x1_hat.T*x1_hat/(x1_hat*x1_hat.T)
    I = np.eye(n)
    y = ((I-P)*x2.T).T
    y_hat = normalize(y)
    z = y_hat+(1/np.tan(np.arccos(rho)))*x1_hat
    z = normalize(z)
    return z

def simu_alpha(conn, code_list, start='2016-01-01', end='2017-01-01', 
                   backfill=True, stocks_price_old=None, business_calendar=None,
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
    if (emu==np.nan).any():
        raise ValueError('There is return missing')
    z = simu_res_alpha(emu, 0.1)
    z = pd.Series(np.reshape(np.array(z), z.shape[1]).tolist(), index=rets.columns)
    return z

def screens_alpha(alpha, buy_pct, hold_pct):
    alpha = alpha.sort_values(ascending=False)
    num = np.cumsum(np.round(len(alpha)*np.array([buy_pct, hold_pct])))
    num = [int(a) for a in num]
    buy_list = alpha[0:num[0]].index.tolist()
    hold_list = alpha[num[0]:num[1]].index.tolist()
    sell_list = alpha[num[1]:].index.tolist()
    return buy_list, hold_list, sell_list
    

def screens_weights(code_list,  buy_list, hold_list, sell_list, w0=None):
    if w0 is None:
        curr_list = buy_list + hold_list
    else:
        curr_list = np.array(code_list)[w0>1e-2].tolist()
        curr_list = list((set(curr_list) | set(buy_list)) - set(sell_list))
    weights = np.array(len(code_list)*[0.0])
    weights[np.in1d(code_list, curr_list)] = 1/len(curr_list)        
    return weights

def screens_alpha_weights(code_list, alpha, w0, buy_pct=0.2, hold_pct=0.4):
    buy_list, hold_list, sell_list = screens_alpha(alpha, buy_pct, hold_pct)
    weights = screens_weights(code_list, buy_list, hold_list, sell_list, w0)
    return weights

if __name__ == '__main__':    
    n = 20
    rho = 0.6
    x1 = np.matrix(1+np.random.randn(n))
    x2 = np.matrix(2+0.5*np.random.randn(n))
    x1_hat = normalize(x1)
    
    P = x1_hat.T*x1_hat/(x1_hat*x1_hat.T)
    I = np.eye(n)
    y = ((I-P)*x2.T).T
    y_hat = normalize(y)
    z = y_hat+(1/np.tan(np.arccos(rho)))*x1_hat
    np.corrcoef(x1, z)
    np.corrcoef(x1_hat, z)
