# -*- coding: utf-8 -*-
"""
Created on Thu May  3 23:17:44 2018

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
    data_path = path + '\\Factor_Model'
elif path.split('/')[-1] == 'Working_On':
    dbpath = os.path.abspath('../../')+'/Data/data.db'
    data_path = path + '/Factor_Model'
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


sz50s = ['600000', '600016', '600019', '600028', '600029', '600030', '600036', 
 '600048', '600050', '600104', '600111', '600309', '600340', '600518',
 '600519', '600547', '600606', '600837', '600887', '600958',
 '600999', '601006', '601088', '601166', '601169', '601186', '601211',
 '601288', '601318', '601328', '601336', '601390', '601398',
 '601601', '601628', '601668', '601669', '601688', '601766', '601800',
 '601818', '601857', '601985', '601988', '601989',
 '603993']

code_list = sz50s

capital = 1000000
rf = 0.03
start = pd.to_datetime('2016-01-05')
end = pd.to_datetime('2017-10-01')
delta = pd.Timedelta(60, unit='d')
now = start
backfill = True

#============mu_hist======================
kwargs={'conn':conn, 
        'code_list':code_list,
        'start':(start-(end-start)).strftime('%Y-%m-%d'),
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d'),
        'backfill': False,
        'lookback_win': 30}
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
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d'),
        'backfill': False,
        'lookback_win': 180}
sigma_hist = sigma_method('hist', mv.hist_expect_sigma, kwargs)

##============sigma_barra===================
if 'model_data' not in dir(): 
     [model_data, wls_resid, factor_list2] = FM_opt.load_data(data_path)
kwargs={
        'code_list':code_list,
        'start':start.strftime('%Y-%m-%d'),
        'window':500,
        'model_data':model_data,
        'wls_resid':wls_resid,
        'factor_list2':factor_list2,
        'conn':data_path}
sigma_barra = sigma_method('barra', FM_opt.barra_stk_cov, kwargs)

kwargs={
        'start':start.strftime('%Y-%m-%d'),
        'window':500,
        'model_data':model_data,
        'wls_resid':wls_resid,
        'factor_list2':factor_list2,
        'conn':data_path}
sigma_barra_all = sigma_method('barra_all', FM_opt.barra_stk_cov_all, kwargs)

#===========mu_reverse====================
kwargs={'code_list': code_list,
        'start': start.strftime('%Y-%m-%d'),
        'window': 500, 
        'model_data': model_data, 
        'wls_resid': wls_resid,
        'factor_list2': factor_list2, 
        'data_path': data_path,
        'conn': conn}
mu_reverse = mu_method('reverse', FM_opt.barra_reverse_mu, kwargs)

opt_quadratic_risky = opt_method('opt_quadratic_risky', 
                                 method=mv.opt_quadratic_risky, 
                                 kwargs={'l':4})

opt_quadratic_risky_restricted = opt_method('opt_quadratic_risky_restricted', 
                                 method=mv.opt_quadratic_risky_restricted, 
                                 kwargs={'l':4,
                                         'w0':[1/len(code_list)]*len(code_list)})

mu_method_all = {'0':mu_hist,
                 '1':mu_capm,
                 '2':mu_reverse}
sigma_method_all = {'0': sigma_hist,
                    '1': sigma_barra}
opt_method_all = {}
for i, l in enumerate(np.arange(1,20,1)):
    temp = opt_method('opt_quadratic_risky'+'-'+str(l), 
                      method=mv.opt_quadratic_risky, 
                      kwargs={'l':l})
    opt_method_all[str(3*i)] = temp
    
    temp = opt_method('opt_quadratic_risky1'+'-'+str(l), 
                      method=mv.opt_quadratic_risky, 
                      kwargs={'l':l})
    opt_method_all[str(3*i+1)] = temp
    
    temp = opt_method('opt_quadratic_risky2'+'-'+str(l), 
                      method=mv.opt_quadratic_risky, 
                      kwargs={'l':l})
    opt_method_all[str(3*i+2)] = temp

    
balance = {}
for i, mu_method_test in mu_method_all.items():
    for j, sigma_method_test in sigma_method_all.items():
        for k, opt_method_test in opt_method_all.items():
            try:
                balance['-'.join([i,j,k])] = rolling.rolling(
                              conn, code_list, start, end, 
                              backfill=True, cap=capital, 
                              mu_method=mu_method_test,
                              sigma_method=sigma_method_test,
                              opt_method=opt_method_test)
            except:
                pass