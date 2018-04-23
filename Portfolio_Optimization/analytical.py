# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:54:48 2018

@author: Kai Zheng
"""

import numpy as np
import pandas as pd
from numpy.linalg import pinv

#%%
def form_mat(mu, sigma):
    if not (isinstance(mu, np.ndarray) and isinstance(sigma, np.ndarray)):
        raise ValueError('mu and sigma should be np.ndarray')
    i = np.matrix(np.ones((np.max(mu.shape),1)))
    mu = np.matrix(mu)
    sigma = np.matrix(sigma)
    if mu.shape[1] != 1:
        mu = mu.T
    return i, mu, sigma

def form_wt(weights):
    if not (isinstance(weights, list) or 
        isinstance(weights, np.ndarray) or
        isinstance(weights, pd.Series)):
        raise ValueError('weights should be a list or np.ndarray')
    w = np.matrix(weights)
    if w.shape[1] != 1:
        w = w.T
    return w

def eff_frt(mu0, mu, sigma):
    '''
    calculate the risky asset weights on 
    unconstraint efficient frontier with short-selling
    '''
    i,mu,sigma = form_mat(mu,sigma)    
    a = float(i.T*pinv(sigma)*i)
    b = float(i.T*pinv(sigma)*mu)
    c = float(mu.T*pinv(sigma)*mu)
    
    g = pinv(sigma)*(c*i - b*mu)/(a*c - b**2)
    h = pinv(sigma)*(a*mu - b*i)/(a*c - b**2)
    w = g + h*mu0
    return w

def gmv(mu, sigma):
    '''
    weights of GMV port
    '''
    i,mu,sigma = form_mat(mu,sigma)
    w = pinv(sigma)*i
    w = w/np.sum(w)
    return w

def CML(mu0, mu, sigma, rf=0.01):
    '''
    calculate the risky asset weights on the capital market line
    given target return
    '''
    i,mu,sigma = form_mat(mu,sigma)    
    C = float((mu0-rf)/((mu-rf*i).T * pinv(sigma) * (mu-rf*i)))
    w_R = C*pinv(sigma)*(mu-rf*i)
    return w_R

def mkt_port(mu, sigma, rf=0.01):
    '''
    Market Portfolio
    '''
    i,mu,sigma = form_mat(mu,sigma)
    w_M = pinv(sigma)*(mu-rf*i)
    w_M = w_M/np.sum(w_M)
    r_M = float(mu.T*w_M)
    vol_M = float(np.sqrt(w_M.T*sigma*w_M))
    return w_M, r_M, vol_M

def statistics(weights, mu, sigma, rf=0.01):
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
    
    w = form_wt(weights)
    _,mu,sigma = form_mat(mu, sigma)
    pret = float(w.T*mu)
    pvol = float(np.sqrt(w.T*sigma*w))
    return np.array([pret, pvol, (pret-rf) / pvol])
    
def proportion_R(mu0, mu, sigma, rf=0.01):
    '''
    proportion of risky assets in total assets 
    in order to satisfy the target return
    '''
    _, r_M, _ = mkt_port(mu, sigma, rf)
    _,mu,sigma = form_mat(mu, sigma)
    p = (mu0-rf)/(r_M-rf)
    return p

def statistics_wrf(p, mu, sigma, rf=0.01):
    '''
    predicted return and volatility for portfolio with risk-free asset
    according to proportion on risky asset.
    '''
    w_M, r_M, vol_M = mkt_port(mu, sigma, rf)
    _,mu,sigma = form_mat(mu, sigma)
    pret = rf + p*(r_M - rf)
    pvol = p*vol_M
    return np.array([pret, pvol])

def statistics_wrf2(w_R, mu, sigma, rf=0.01):
    '''
    predicted return and volatility for portfolio with risk-free asset
    according to weights on risky asset.
    '''
    w_M, r_M, vol_M = mkt_port(mu, sigma, rf)
    _,mu,sigma = form_mat(mu, sigma)
    pret = float(w_R.T*mu + (1-w_R.sum())*rf)
    pvol = float(np.sqrt(w_R.T*sigma*w_R))
    return np.array([pret, pvol])

def risk_premium(mu, sigma, rf=0.01):
    '''
    market price of risk
    '''
    w_M, r_M, vol_M = mkt_port(mu, sigma, rf)
    return (r_M-rf)/vol_M

#%%
if __name__ == '__main__':
    mu = np.array([[0.079, 0.079, 0.09, 0.071],])
    std = np.array([[0.195, 0.182, 0.183, 0.165],])
    corr = np.array([[1, 0.24, 0.25, 0.22], 
                     [0.24, 1, 0.47, 0.14], 
                     [0.25, 0.47, 1, 0.25],
                     [0.22, 0.14, 0.25, 1]])
    sigma = np.multiply(np.dot(std.T, std), corr)
    
    mu0 = 0.04
    w_R = CML(mu0, mu, sigma)
    w_R
    p = proportion_R(mu0, mu, sigma)
    p
    pret, pvol = statistics_wrf(p, mu, sigma)
    pret
    pvol
    pret, pvol = statistics_wrf2(w_R, mu, sigma)
    pret
    pvol
    w = eff_frt(mu0, mu, sigma)
    w
    pret, pvol, sharpe = statistics(w_R, mu, sigma)
    pret
    pvol
    sharpe
