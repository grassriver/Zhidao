# -*- coding: utf-8 -*-
"""
Rolling Optimization

Created on Sat Apr  7 16:06:00 2018

@author: Kai Zheng
"""

import numpy as np
import pandas as pd
import general_tools as tool
import sqlite3 as sql
import stock_screener as sc
from portfolio_class import Portfolio
from stock_class import Stock
import Portfolio_Optimization.mv as mv
#import Portfolio_Optimization.analytical as ant
#import CAPM.capm as capm

# =============================================================================
# input: stock_list
# output: portfolio balance
# =============================================================================

#%%
# forward fill
def rolling(conn, stock_list, start, end, step='M', cap=1000000, 
            backfill=False, mu_method=None, sigma_method=None, opt_method=None):
    delta = pd.Timedelta(1, unit=step)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    now = start
    stocks_price_old = tool.get_stocks_price_old(conn, stock_list)
    business_calendar = tool.get_business_calendar(conn)
    industry = tool.get_industry(conn, stock_list)
    benchmark_old = tool.get_benchmark_old(conn)
    if mu_method is not None:
        mu_method.update(stocks_price_old=stocks_price_old,
                         business_calendar=business_calendar,
                         industry=industry,
                         backfill=backfill)
    if sigma_method is not None:
        sigma_method.update(stocks_price_old=stocks_price_old,
                            business_calendar=business_calendar,
                            industry=industry,
                            backfill=backfill)
    balance = pd.Series()
    while now < end:
# =============================================================================
#         update methods parameters
#         mu_method = update_method()
#         sigma_method = update_sigma()
#         opt_method = update_opt()
# =============================================================================
        (balance, 
         cap, 
         mk_port, 
         weights) = single_stage_opt(
                         conn, stock_list, now, 
                         delta, cap, balance, backfill, 
                         mu_method=mu_method,
                         sigma_method=sigma_method,
                         opt_method=opt_method,
                         stocks_price_old=stocks_price_old,
                         business_calendar=business_calendar,
                         industry=industry,
                         benchmark_old=benchmark_old)
        now = now + delta
        if now + delta > end:
            delta = end - now
    return balance

def single_stage_opt(conn, stock_list, now, delta, cap, balance, backfill,
                     mu_method=None, sigma_method=None, opt_method=None, 
                     stocks_price_old=None, business_calendar=None,
                     industry=None, benchmark_old=None):
    #================ check date availability===========================
    now = pd.to_datetime(
            tool.first_trading_day(conn, now.strftime('%Y-%m-%d'), 
                                         business_calendar=business_calendar))    
    #================ predict mean anv cov of returns===================

    # estimate mean of returns
    if mu_method is None:
        mu = mv.hist_expect_mu(
                    conn, stock_list,
                    start=(now-delta).strftime('%Y-%m-%d'),
                    end=(now).strftime('%Y-%m-%d'),
                    backfill = backfill, 
                    stocks_price_old=stocks_price_old,
                    business_calendar=business_calendar,
                    industry=industry)
    elif mu_method.name == 'capm':
        mu_method.update(start=now.strftime('%Y-%m-%d'))
        mu = mu_method.run()
    elif mu_method.name == 'hist':
        mu_method.update(start = (now-delta).strftime('%Y-%m-%d'),
                         end = (now-pd.Timedelta(1)).strftime('%Y-%m-%d'))
        mu = mu_method.run()
    else:
        raise ValueError('mu_method not right!')
    
 
    # estimate covariance matrix of returns
    if sigma_method is None:
        sigma = mv.hist_expect_sigma(
                    conn, stock_list,
                    start=(now-delta).strftime('%Y-%m-%d'),
                    end=(now).strftime('%Y-%m-%d'),
                    backfill = backfill, 
                    stocks_price_old=stocks_price_old,
                    business_calendar=business_calendar,
                    industry=industry)
    elif sigma_method.name == 'hist':
        sigma_method.update(
                start = (now-delta).strftime('%Y-%m-%d'),
                end = (now-pd.Timedelta(1)).strftime('%Y-%m-%d'))
        sigma = sigma_method.run()
    else:
        raise ValueError('sigma_method not right!')
        
    #================= mean-variance optimization ======================
    if opt_method is None:
        # use numeric solution
        opts, optv= mv.opt_s_v(mu,sigma)
        weights = opts.x
    elif opt_method.name in ['opt_quadratic', 'opt_s', 'opt_v']:
        opt_method.update(mu=mu, sigma=sigma)
        weights = opt_method.run()
    elif opt_method.name == 'restricted':
        weights = opt_method.run()
    else:
        raise ValueError('opt_method not right!')
        
    if np.isnan(weights).any():
        print(mu)
        print(sigma)
        raise ValueError('optimal weights error')
#    # use analytical solution
#    mu = np.array(mu)
#    sigma = np.array(sigma)
#    if goal == 's':
#        weights, _, _ = ant.mkt_port(mu, sigma)
#    elif goal == 'v':
#        weights = ant.gmv(mu, sigma)
#    else:
#        raise ValueError('provide the correct goal.')
#    weights = np.array(weights)[:,0]
    print(weights)
    
    #================== backtest ==========================
    
    # calculate shares of each stock according to weights
    if backfill==False:
        mport = tool.portfolio_construct_by_weight(conn, 
                                                   now.strftime('%Y-%m-%d'), 
                                                   pd.Series(stock_list), 
                                                   weights = weights,
                                                   capital = cap)
    else:
        stocks = Stock(conn, stock_list, 
                       start=now.strftime('%Y-%m-%d'), 
                       end=(now+delta).strftime('%Y-%m-%d'),
                       backfill = backfill, 
                       stocks_price_old=stocks_price_old,
                       business_calendar=business_calendar,
                       industry=industry)
        date = stocks.start
        stock_price = stocks.price[date].reset_index()
        shares = np.floor(cap*weights/stock_price['close']/100)*100 
        mport = pd.DataFrame({'code': stock_list, 'shares': shares})
    
    # construct a portfolio according to mport
    mk_port = Portfolio(
                    conn, mport, 
                    start=now.strftime('%Y-%m-%d'), 
                    end=(now+delta).strftime('%Y-%m-%d'), 
                    backfill=backfill,
                    stocks_price_old = stocks_price_old,
                    business_calendar=business_calendar,
                    industry=industry,
                    benchmark_old=benchmark_old)
    # update balance                        
    balance = balance.append(mk_port.port_daily_balance())
    cap = balance[-1]
    return balance, cap, mk_port, weights


if __name__ == '__main__':
    conn_path = 'D:/Kaizheng/Working_directory/portfolio_intelligence/PI/Data/data.db'
    conn = sql.connect(conn_path)
    date = '2015-10-31'
    start='2016-01-05'
    end='2016-10-31'

    stock_list = sc.stock_screener_ranking(conn_path, date=date, 
                                           var_list=['net_profit_ratio','roe','eps'],
                                           rank_by = 'roe',order = 'ascending',top=10)
    stocks = tool.portfolio_construct(conn,start = '2016-01-05',code_list=stock_list.Code,
                                      construct_type='weight',equal=True)
    code_list = list(stocks.code)

    # Plot Efficient Frontier
    mu,sigma = mv.hist_expect(conn, code_list, start=start, end=end) #cheating here
    mv.plot_eff_fter(mu, sigma)
    weights = mv.tan_port(mu,sigma).x

    balance = rolling(conn, code_list, start, end)
    print(balance)
    balance.plot()