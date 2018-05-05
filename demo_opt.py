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
if path.split('\\')[-1] == 'Working_On':
    dbpath = os.path.abspath('../../')+'\\Data\\data.db'
elif path.split('/')[-1] == 'Working_On':
    dbpath = os.path.abspath('../../')+'/Data/data.db'
else:
    raise ValueError('enter Working_On path!')
sys.path.append(path)
conn = sql.connect(dbpath)

from stock_class import Stock
from portfolio_class import Portfolio
from Portfolio_Optimization import mv
import Portfolio_Optimization.rolling as rolling
import stock_screener as sc
import general_tools as tool
import CAPM.capm as capm
from Portfolio_Optimization.methodclass import mu_method, sigma_method, opt_method
import Factor_Model.FM_for_Optimization as FM_opt

#%%--------------rolling-------------------------------
start = pd.to_datetime('2016-06-03')
end = pd.to_datetime('2016-12-01')
code_list = ['000001', '000002', '000004', '000005', '000006', '000007', 
             '002415', '600298', '002736', '002230', '600519', '000651']

#sz50s = ['600000', '600016', '600019', '600028', '600029', '600030', '600036', 
# '600048', '600050', '600104', '600111', '600309', '600340', '600518',
# '600519', '600547', '600606', '600837', '600887', '600919', '600958',
# '600999', '601006', '601088', '601166', '601169', '601186', '601211',
# '601229', '601288', '601318', '601328', '601336', '601390', '601398',
# '601601', '601628', '601668', '601669', '601688', '601766', '601800',
# '601818', '601857', '601878', '601881', '601985', '601988', '601989',
# '603993']
#
#code_list = sz50s

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

#============mu_capm=====================
kwargs={'conn': conn, 
        'start': (pd.to_datetime(start)-pd.Timedelta(1)).strftime('%Y-%m-%d'), 
        'lookback_win': 252, 
        'stk_list': code_list,
        'proj_period':30, 
        'proj_method':'arma_garch',
        'freq': 'daily'}
mu_capm = mu_method('capm', capm.capm_mu, kwargs)


#============sigma_hist====================
kwargs={'conn':conn, 
        'code_list':code_list,
        'start':(start-(end-start)).strftime('%Y-%m-%d'),
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d')}
sigma_hist = sigma_method('hist', mv.hist_expect_sigma, kwargs)

#========opt_quadratic==============
opt_quadratic = opt_method('opt_quadratic', method=mv.opt_quadratic, 
                           kwargs={'l':4})

opt_quadratic_risky = opt_method('opt_quadratic_risky', method=mv.opt_quadratic_risky, 
                           kwargs={'l':4})

#==============rolling optimization=============================
balance1 = rolling.rolling(
              conn, code_list, start, end, 
              backfill=True, cap=capital, 
              mu_method=mu_hist,
              sigma_method=sigma_hist,
              opt_method=opt_quadratic_risky)

# benchmark performance
benchmark = 'sh000001'
query = 'select date, close from index_price where code = "' + benchmark + '"'
bench_price = pd.read_sql(query, conn)
bench_price['date'] = pd.to_datetime(bench_price['date'])
bench_price.set_index('date', inplace=True)

# equal weight portfolio
stocks = tool.portfolio_construct_by_weight(
            conn, start, code_list,
            capital=capital, backfill=True)
ewet_port = Portfolio(conn, stocks, start=start, end=end, backfill=True)
ewet_port_balance = ewet_port.port_daily_balance()

# combine together
balance1 = capital/balance1[0]*balance1
ewet_port_balance = capital/ewet_port_balance[0]*ewet_port_balance

df = pd.merge(pd.DataFrame(balance1), bench_price,
              left_index=True, right_index=True, how = 'left')
#df = df.join(pd.DataFrame(balance2, columns=['restrictd']))
#df = df.join(pd.DataFrame(balance3, columns=['capm']))
df = df.join(pd.DataFrame({'ewet':ewet_port_balance}))
#df.columns=['portfolio', 'benchmark', 'restricted', 'capm', 'equal weight portfolio']
#df.columns=['portfolio', 'benchmark', 'restricted', 'equal weight portfolio']
df.columns=['portfolio', 'benchmark', 'equal weight portfolio']
df['benchmark'] = capital/df['benchmark'][0]*df['benchmark']
df.plot(figsize=(16,9))



#%% test risk free 
mu = np.array([0.079, 0.079, 0.09, 0.071])
std = np.array([[0.195, 0.182, 0.183, 0.165],])
corr = np.array([[1, 0.24, 0.25, 0.22], 
                 [0.24, 1, 0.47, 0.14], 
                 [0.25, 0.47, 1, 0.25],
                 [0.22, 0.14, 0.25, 1]])
sigma = np.multiply(np.dot(std.T, std), corr)

weights = mv.opt_quadratic_risky(mu, sigma, 4)


#%% test FM
start = pd.to_datetime('2016-11-03')
end = pd.to_datetime('2017-06-01')
#code_list = ['000001', '000002', '000004', '000005', '000006', '000007', 
#             '002415', '600298', '002736', '002230', '600519', '000651']

data_path = path + '\\Factor_Model'
[model_data, wls_resid, factor_list2] = FM_opt.load_data(data_path)

#start = '2017-01-05'
#window = 500
#code_list = ['600825','600229','000001','600743','600795','600108']
#PreRetCov = FM_opt.barra_stk_cov(code_list, start, window, model_data, 
#                                 wls_resid, factor_list2, data_path)

sz50s = ['600000', '600016', '600019', '600028', '600029', '600030', '600036', 
 '600048', '600050', '600104', '600111', '600309', '600340', '600518',
 '600519', '600547', '600606', '600837', '600887', '600958',
 '600999', '601006', '601088', '601166', '601169', '601186', '601211',
 '601288', '601318', '601328', '601336', '601390', '601398',
 '601601', '601628', '601668', '601669', '601688', '601766', '601800',
 '601818', '601857', '601985', '601988', '601989',
 '603993']

code_list = sz50s

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

#============mu_capm=====================
kwargs={'conn': conn, 
        'start': (pd.to_datetime(start)-pd.Timedelta(1)).strftime('%Y-%m-%d'), 
        'lookback_win': 252, 
        'stk_list': code_list,
        'proj_period':30, 
        'proj_method':'arma_garch',
        'freq': 'daily'}
mu_capm = mu_method('capm', capm.capm_mu, kwargs)


#============sigma_hist====================
kwargs={'conn':conn, 
        'code_list':code_list,
        'start':(start-(end-start)).strftime('%Y-%m-%d'),
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d')}
sigma_hist = sigma_method('hist', mv.hist_expect_sigma, kwargs)

#============sigma_barra===================
kwargs={
        'code_list':code_list,
        'start':start.strftime('%Y-%m-%d'),
        'window':500,
        'model_data':model_data,
        'wls_resid':wls_resid,
        'factor_list2':factor_list2,
        'conn':data_path}
sigma_barra = sigma_method('barra', FM_opt.barra_stk_cov, kwargs)

opt_quadratic_risky = opt_method('opt_quadratic_risky', 
                                 method=mv.opt_quadratic_risky, 
                                 kwargs={'l':20})

opt_quadratic_risky_restricted = opt_method('opt_quadratic_risky_restricted', 
                                 method=mv.opt_quadratic_risky_restricted, 
                                 kwargs={'l':20,
                                         'w0':[1/len(code_list)]*len(code_list)})

balance1 = rolling.rolling(
              conn, code_list, start, end, 
              backfill=True, cap=capital, 
              mu_method=mu_hist,
              sigma_method=sigma_barra,
              opt_method=opt_quadratic_risky)

balance2 = rolling.rolling(
              conn, code_list, start, end, 
              backfill=True, cap=capital, 
              mu_method=mu_hist,
              sigma_method=sigma_hist,
              opt_method=opt_quadratic_risky)

balance3 = rolling.rolling(
              conn, code_list, start, end, 
              backfill=True, cap=capital, 
              mu_method=mu_hist,
              sigma_method=sigma_barra,
              opt_method=opt_quadratic_risky_restricted)

# benchmark performance
benchmark = 'sh000016'
query = 'select date, close from index_price where code = "' + benchmark + '"'
bench_price = pd.read_sql(query, conn)
bench_price['date'] = pd.to_datetime(bench_price['date'])
bench_price.set_index('date', inplace=True)

# equal weight portfolio
stocks = tool.portfolio_construct_by_weight(
            conn, start, code_list,
            capital=capital, backfill=True)
ewet_port = Portfolio(conn, stocks, start=start, end=end, backfill=True)
ewet_port_balance = ewet_port.port_daily_balance()

# combine together
balance1 = capital/balance1[0]*balance1
ewet_port_balance = capital/ewet_port_balance[0]*ewet_port_balance

df = pd.merge(pd.DataFrame(balance1), bench_price,
              left_index=True, right_index=True, how = 'left')
df = df.join(pd.DataFrame(balance2, columns=['hist_predict']))
df = df.join(pd.DataFrame(balance3, columns=['test']))
df = df.join(pd.DataFrame({'ewet':ewet_port_balance}))
#df.columns=['portfolio', 'benchmark', 'restricted', 'capm', 'equal weight portfolio']
#df.columns=['portfolio', 'benchmark', 'restricted', 'equal weight portfolio']
df.columns=['barra', 'benchmark','hist', 'barra restricted', 'equal weight portfolio']
df['benchmark'] = capital/df['benchmark'][0]*df['benchmark']
df.plot(figsize=(16,9))
