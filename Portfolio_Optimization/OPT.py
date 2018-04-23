# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:43:50 2018

@author: Kai Zheng
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import Portfolio_Optimization.mv as mv
import general_tools as tool
from portfolio_class import Portfolio
import Portfolio_Optimization.rolling as rolling

#%% backfilled one stage optimization
def OPT_ONE(code_list, conn, start, end, capital, backfill=True):
    # Plot Efficient Frontier
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    mu, sigma = mv.hist_expect(conn, code_list,
                               start=(start-(end-start)).strftime('%Y-%m-%d'),
                               end=(start-pd.Timedelta(1)).strftime('%Y-%m-%d'))  # cheating here
    print('Derive expected return and covariance matrix from {0} to {1}'.\
          format((start-(end-start)).strftime('%Y-%m-%d'),
                 (start-pd.Timedelta(1)).strftime('%Y-%m-%d')))
    mv.plot_eff_fter(mu, sigma)
    opts, optv = mv.opt_s_v(mu, sigma)
    weights = opts.x
    
    # Portfolio Comparison
    mport = tool.portfolio_construct_by_weight(conn, start, code_list,
                                               weights=weights, capital=capital, backfill=True)
    mk_port = Portfolio(conn, mport, start=start, end=end, backfill=True)
    
    stocks = tool.portfolio_construct_by_weight(conn, start, code_list,
                                                capital=capital, weights = None, backfill=True)
    ewet_port = Portfolio(conn, stocks, start=start, end=end, backfill=True)
    
    dfs = {'optimized portfolio': mk_port.port_daily_balance(), # series
           'benchmark': mk_port.benchmark()['price'], # dataframe
           'equal weight portfolio': ewet_port.port_daily_balance(), # series
           }
    df = pd.DataFrame(dfs)
    df = df/df.iloc[0,:]*capital #normalize
    df.plot()
    
# rolling optimization
def OPT_ROLL(code_list, conn, start, end, capital, backfill=True):
    balance = rolling.rolling(conn, code_list, start, end, goal='s', backfill=True, cap=capital)
    #balance.plot()
    
    # benchmark performance
    benchmark = 'sh000001'
    query = 'select date, close from index_price where code = "' + benchmark + '"'
    bench_price = pd.read_sql(query, conn)
    bench_price['date'] = pd.to_datetime(bench_price['date'])
    bench_price.set_index('date', inplace=True)
    
    # equal weight portfolio
    stocks = tool.portfolio_construct_by_weight(conn, start, code_list,
                                                capital=capital, backfill=True)
    ewet_port = Portfolio(conn, stocks, start=start, end=end, backfill=True)
    #ewet_port.port_performance_matrix()
    #ewet_port.nav_plot()
    #ewet_port.port_summary()
    #ewet_port._code_list
    ewet_port_balance = ewet_port.port_daily_balance()
    
    # combine together
    balance = capital/balance[0]*balance
    ewet_port_balance = capital/ewet_port_balance[0]*ewet_port_balance
    
    df = pd.merge(pd.DataFrame(balance), bench_price,
                  left_index=True, right_index=True, how = 'left')
    df = df.join(pd.DataFrame({'ewet':ewet_port_balance}))
    df.columns=['portfolio', 'benchmark', 'equal weight portfolio']
    df['benchmark'] = capital/df['benchmark'][0]*df['benchmark']
    df.plot()
    
if __name__ == '__main__':
    import sqlite3 as sql
    import matplotlib.pyplot as plt
    conn = sql.connect('../../../Data/data.db')
    start = '2017-01-05'
    end = '2017-10-01'
    code_list = ['000001', '000002', '000004', '000005', '000006', '000007']
    capital = 1000000
    OPT_ONE(code_list, conn, start, end, capital)
    plt.show()
    OPT_ROLL(code_list, conn, start, end , capital)
    plt.show()