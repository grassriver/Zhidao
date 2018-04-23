# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 23:43:16 2018

@author: Kai Zheng
"""

import pandas as pd
import numpy as np
import sqlite3 as sql
import sys
import os

path = os.path.abspath('./')
if path.split('\\')[-1] != 'Working_On':
    raise ValueError('enter working on path!')
sys.path.append(path)
dbpath = os.path.abspath('../../')+'\\Data\\data.db'
conn = sql.connect(dbpath)

from stock_class import Stock
from portfolio_class import Portfolio
from Portfolio_Optimization import mv
import Portfolio_Optimization.rolling as rolling
import stock_screener as sc
import general_tools as tool
import CAPM.capm as capm
from Portfolio_Optimization.methodclass import mu_method, sigma_method, opt_method


#%%--------------rolling-------------------------------
start = pd.to_datetime('2017-01-03')
end = pd.to_datetime('2017-06-01')
code_list = ['000001', '000002', '000004', '000005', '000006', '000007', 
             '002415', '600298', '002736', '002230', '600519']
capital = 1000000
kwargs={'conn': conn, 
        'start': (pd.to_datetime(start)-pd.Timedelta(1)).strftime('%Y-%m-%d'), 
        'lookback_win': 252, 
        'stk_list': code_list,
        'proj_period':30, 
        'proj_method':'arma_garch',
        'freq': 'daily'}
#stk_prediction = capm.capm_mu(**kwargs)
#params = {'name': 'capm', 'method': capm.capm_mu, 'kwargs': kwargs}
mu_capm = mu_method('capm', capm.capm_mu, kwargs)

#============mu_hist======================
kwargs={'conn':conn, 
        'code_list':code_list,
        'start':(start-(end-start)).strftime('%Y-%m-%d'),
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d')}
mu_hist = mu_method('hist', mv.hist_expect_mu, kwargs)


#============sigma_hist====================
kwargs={'conn':conn, 
        'code_list':code_list,
        'start':(start-(end-start)).strftime('%Y-%m-%d'),
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d')}
sigma_hist = sigma_method('hist', mv.hist_expect_sigma, kwargs)

#========opt_quadratic==============
opt_quadratic = opt_method('opt_quadratic', method=mv.opt_quadratic, 
                           kwargs={'l':3})

#==============rolling optimization=============================
balance = rolling.rolling(conn, code_list, start, end, 
                          backfill=True, cap=capital, 
                          mu_method=mu_hist,
                          sigma_method=sigma_hist,
                          opt_method=opt_quadratic)

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
ewet_port_balance = ewet_port.port_daily_balance()

# combine together
balance = capital/balance[0]*balance
ewet_port_balance = capital/ewet_port_balance[0]*ewet_port_balance

df = pd.merge(pd.DataFrame(balance), bench_price,
              left_index=True, right_index=True, how = 'left')
#df = df.join(pd.DataFrame(balance2, columns=['restrictd']))
#df = df.join(pd.DataFrame(balance3, columns=['capm']))
df = df.join(pd.DataFrame({'ewet':ewet_port_balance}))
#df.columns=['portfolio', 'benchmark', 'restricted', 'capm', 'equal weight portfolio']
#df.columns=['portfolio', 'benchmark', 'restricted', 'equal weight portfolio']
df.columns=['portfolio', 'benchmark', 'equal weight portfolio']
df['benchmark'] = capital/df['benchmark'][0]*df['benchmark']
df.plot(figsize=(16,9))
