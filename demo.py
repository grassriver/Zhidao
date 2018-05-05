# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:20:06 2018

@author: Kai Zheng
"""

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np
import sqlite3 as sql
import sys
import os

path = os.path.abspath('./')
if path.split('\\')[-1] != 'Working_On':
    raise ValueError('enter Working_On path!')
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
from Portfolio_Optimization.methodclass import mu_method

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
# Pre-defined stock list
start = '2017-01-05'
end = '2017-10-01'
code_list = ['000001', '000002', '000004', '000005', '000006', '000007']
stocks = tool.portfolio_construct(conn, start=start, code_list=code_list,
                                  construct_type='weight', equal=True)
code_list = list(stocks.code)

mu, sigma = mv.hist_expect(conn, code_list, start=start, end=end)
mv.plot_eff_fter(mu, sigma)

ewet_port = Portfolio(conn, stocks, start=start, end=end)
ewet_port.port_summary()

#%% Demo for whole process
# Stock Screening
conn_path = 'D:/Kaizheng/Working_directory/portfolio_intelligence/PI/Data/data.db'
date = '2016-10-31'
start = '2017-01-05'
end = '2017-10-31'

stock_list = sc.stock_screener_ranking(conn_path, date=date,
                                       var_list=['net_profit_ratio', 'roe', 'eps'],
                                       rank_by='roe', order='descending', top=10)
stocks = tool.portfolio_construct(conn, start='2016-01-05', code_list=stock_list.Code,
                                  construct_type='weight', equal=True)
code_list = list(stocks.code)

# Plot Efficient Frontier
mu, sigma = mv.hist_expect(conn, code_list, start='2015-10-01', end='2015-12-31')  # cheating here
mv.plot_eff_fter(mu, sigma)
opts, optv = mv.opt_s_v(mu, sigma)
weights = opts.x

# Portfolio Comparison
mport = tool.portfolio_construct_by_weight(conn, '2016-01-05', pd.Series(code_list), weights=weights)
mk_port = Portfolio(conn, mport, start=start, end=end)
mk_port.port_performance_matrix()
mk_port.nav_plot()
mk_port.port_summary()

ewet_port = Portfolio(conn, stocks, start=start, end=end)
ewet_port.port_performance_matrix()
ewet_port.nav_plot()
ewet_port.port_summary()

mv.max_sharpe(mu, sigma)
mv.min_vol(mu, sigma)
mv.max_ret_st_vol(0.50, mu, sigma)
mv.min_vol_st_ret(0.08, mu, sigma)
mv.max_ret_st_vol_wrf(0.13, mu, sigma)
mv.min_vol_st_ret_wrf(0.08, mu, sigma)

#%% backfilled one stage optimization
start = '2017-01-05'
end = '2017-10-01'
code_list = ['000001', '000002', '000004', '000005', '000006', '000007']
capital = 1000000

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


#%%==============rolling===================================
start = '2017-01-06'
end = '2017-10-01'
code_list = ['000001', '000002', '000004', '000005', '000006', '000007', 
             '002415', '002635', '600298', '002736', '601111', '002230']

# rolling optimization
capital = 1000000
balance = rolling.rolling(conn, code_list, start, end, goal='s', backfill=True, cap=capital)
balance2 =  rolling.rolling_restricted(conn, code_list, start, end, goal='s', backfill=True, cap=capital)
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
df = df.join(pd.DataFrame(balance2, columns=['restrictd']))
df = df.join(pd.DataFrame({'ewet':ewet_port_balance}))
df.columns=['portfolio', 'benchmark', 'restricted', 'equal weight portfolio']
df['benchmark'] = capital/df['benchmark'][0]*df['benchmark']
df.plot()

#%%

start = '2016-07-07'
end = '2016-12-31'
capital = 1000000
code_list = ['000001', '000002', '000005', '000006', 
             '002415', '002635', '600298', '002736', '601111', '002230']


stk_prediction = capm.capm_mu(conn, '2016-07-06', 252, stk_list=code_list, 
                              freq='daily', proj_method='arma_garch', 
                              arma_order=[0,0,2], garch_order=[1,1], proj_period=60)

# Plot Efficient Frontier
start = pd.to_datetime(start)
end = pd.to_datetime(end)

mu, sigma = mv.hist_expect(conn, code_list,
                           start=(start-(end-start)).strftime('%Y-%m-%d'),
                           end=(start-pd.Timedelta(1)).strftime('%Y-%m-%d'))
#mu, sigma = mv.hist_expect(conn, code_list,
#                           start = start.strftime('%Y-%m-%d'),
#                           end = end.strftime('%Y-%m-%d'))  # cheating here

if input('capm? y or n \n')=='y':
    mu = stk_prediction

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


#%%--------------capm rolling-------------------------------
start = '2014-01-03'
end = '2017-06-01'
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

#==============rolling optimization=============================
balance = rolling.rolling(conn, code_list, start, end, 
                          backfill=True, cap=capital)

balance2 = rolling.rolling_restricted(conn, code_list, start, end, 
                                      backfill=True, cap=capital)

balance3 = rolling.rolling(conn, code_list, start, end, goal='s', 
                           backfill=True, cap=capital, mu_method=mu_capm)

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
df = df.join(pd.DataFrame(balance2, columns=['restrictd']))
#df = df.join(pd.DataFrame(balance3, columns=['capm']))
df = df.join(pd.DataFrame({'ewet':ewet_port_balance}))
#df.columns=['portfolio', 'benchmark', 'restricted', 'capm', 'equal weight portfolio']
df.columns=['portfolio', 'benchmark', 'restricted', 'equal weight portfolio']
df['benchmark'] = capital/df['benchmark'][0]*df['benchmark']
df.plot(figsize=(16,9))


#%% ------------reverse optimization------------------------------
sigma = pd.read_pickle('./Factor model/FM_PreRetCov_2017_01_03.pkl')
sigma = sigma.sort_values('code')
sigma = sigma.reindex_axis(sorted(sigma.columns), axis=1)
code_list_all = list(sigma.code)
sigma = sigma.set_index('code').sort_index()
sigma_all = np.matrix(sigma)

query1 = 'select * from stocks_price where date = "{}"'.format('2017-01-03')
stocks = pd.read_sql(query1, conn)
stocks = stocks[['code', 'close']].sort_values('code')
stocks = stocks.loc[np.in1d(stocks.code, code_list_all), :].sort_values('code')

query2 = 'select code, outstanding from stock_basics'
outstanding = pd.read_sql(query2, conn).sort_values('code')

stocks = pd.merge(stocks, outstanding, on='code', how='left')
stocks['cap'] = stocks['close']*stocks['outstanding']
stocks['weight'] = stocks['outstanding']/stocks['outstanding'].sum()

w = np.mat(stocks.weight).T
delta = 0.5/np.sqrt(w.T*sigma_all*w)[0,0]
mu = delta*sigma_all*w
mu_all = pd.DataFrame({'code':code_list_all, 'mu':mu.reshape((len(mu),)).tolist()[0]})


start = '2017-01-03'
end = '2017-03-01'
capital = 1000000
code_list = ['000001', '000002', '000004', '000005', '000006', '000007', 
             '002415', '600298', '002736', '002230', '600519']

code_list = np.array(code_list_all)[np.unique(np.random.randint(0,len(code_list_all),50))].tolist()

mu = (mu_all.loc[np.isin(mu_all['code'], code_list),'mu']*252).tolist()
sigma = sigma_all[np.isin(code_list_all, code_list)][:,np.isin(code_list_all, code_list)]*252

#stk_prediction = capm.capm_mu(conn, '2016-12-30', 252, stk_list=code_list, 
#                              freq='daily', proj_method='arma_garch', 
#                              arma_order=[0,0,2], garch_order=[1,1], proj_period=60)
#mu = stk_prediction
#sigma = sigma_all[np.isin(code_list_all, code_list)][:,np.isin(code_list_all, code_list)]*252

#mv.plot_eff_fter(mu, sigma)
#opts, optv = mv.opt_s_v(mu, sigma)
#weights = mv.max_ret_st_vol(0.04, mu, sigma)
weights = mv.opt_quadratic(mu, sigma, 10)

opt_method = method('opt_quadratic', method=mv.opt_quadratic, 
                    kwargs={'mu':mu, 'sigma':sigma, 'l':10})
opt_method.run()

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
np.std(weights)