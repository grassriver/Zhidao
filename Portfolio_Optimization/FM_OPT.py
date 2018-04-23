# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:21:56 2018

@author: Kai Zheng
"""

import numpy as np
import pandas as pd
import sqlite3 as sql
import sys
sys.path.append('..')
import Portfolio_Optimization.mv as mv
import Portfolio_Optimization.rolling as rolling
import general_tools as tool
from portfolio_class import Portfolio
from stock_class import Stock

conn = sql.connect('D:/Kaizheng/Working_directory/portfolio_intelligence/PI/Data/data.db')

MU = pd.read_pickle('D:/Kaizheng/Working_directory/portfolio_intelligence/PI/Code/Working_On/Factor model/test_FM_PreReturn.pkl')
SIGMA = pd.read_pickle('D:/Kaizheng/Working_directory/portfolio_intelligence/PI/Code/Working_On/Factor model/test_FM_PreRetCov.pkl')

MU = MU[['code', 'PreRet']].groupby('code').mean()*252

code_list = ['000001', '000002', '000004', '000005', '000006', '000007']
        
#%% backfilled one stage optimization
start = '2017-01-05'
end = '2017-10-01'
code_list = ['000001', '000002', '000004', '000005', '000006', '000007']
capital = 1000000

# Plot Efficient Frontier
start = pd.to_datetime(start)
end = pd.to_datetime(end)

#mu, sigma = mv.hist_expect(conn, code_list,
#                           start=(start-(end-start)).strftime('%Y-%m-%d'),
#                           end=(start-pd.Timedelta(1)).strftime('%Y-%m-%d'))  # cheating here
#print('Derive expected return and covariance matrix from {0} to {1}'.\
#      format((start-(end-start)).strftime('%Y-%m-%d'),
#             (start-pd.Timedelta(1)).strftime('%Y-%m-%d')))

n = len(code_list)
sigma = np.zeros([len(code_list), len(code_list)])
mu = np.zeros([n,])
for i in range(n):
    mu[i] = MU.loc[code_list[i], 'PreRet']
    for j in range(n):
        sigma[i,j] = SIGMA.loc[SIGMA['code']==code_list[i], code_list[j]]*252

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

