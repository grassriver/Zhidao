# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:20:06 2018

@author: Kai Zheng
"""

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import pandas as pd
import sqlite3 as sql
import sys

user = input('Please enter your name: ')
if user=='kai':
    sys.path.append('d:/Kaizheng/Working_directory/portfolio_intelligence/PI/Code/Working_On')
    conn = sql.connect("d:/Kaizheng/Working_directory/portfolio_intelligence/PI/data/data.db")
elif user =='hillary':
    sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On')
    conn = sql.connect("C:/Users/Hillary/Documents/PI/data/data.db")
else:
    raise ValueError('Please enter correct user name. You could add an elif condition for your own local setting.')

from stock_class import Stock
from portfolio_class import Portfolio

#%% Demo Stock Class
code_list = ['000001', '000002']
s = Stock(conn, code_list, start='2017-01-05', end='2017-02-01')
s.code
s.price
s.close_price
s.daily_returns
s.daily_cum_returns

#%% Demo Portfolio Class
stocks_list = pd.DataFrame({'code': ['000002', '000001'], 'shares': [1000, 100]})
p = Portfolio(conn, stocks_list, start='2017-01-05', end='2017-02-01')
p.stock_price()
p.stock_returns()
p.weekly_returns()
p.monthly_returns()
p.port_returns()
p.benchmark()
p.benchmark_info
p.benchmark_returns()
p.add_benchmark(index_code='sh000002')
p.stock_daily_balance()
p.performance_matrix()
p.port_performance_matrix()
p.port_summary()
p.port_initial_shares()
p.port_daily_balance()
p.port_allocation()
p.allocation_plot()
p.port_balance_plot()
p.gen_drawdown_table()
p.plot_drawdown_periods()
p.plot_drawdown_underwater()
p.candle_stick_plot('000001')


#%% Portfolio Optimization
import Portfolio_Optimization.mv as mv
code_list = ['000001', '000002', '000004', '000005', '000006', '000007']
s = Stock(conn, code_list, start='2016-01-01', end='2017-01-01')
price = pd.pivot_table(s.price, values='close', index=s.price.index, columns='code').fillna(method='bfill')
rets = s.daily_returns
mu = rets.mean()*252
sigma = rets.cov()*252
mv.plot_eff_fter(mu, sigma)
