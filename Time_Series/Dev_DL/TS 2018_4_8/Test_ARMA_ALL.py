# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:45:17 2018

@author: Administrator
"""
"""
Created on 2018-03-13 19:14:30

@author: Daniel Lu
"""
#%%
import pandas as pd
import sqlite3 as sql
import sys
sys.path.append('F:/Nut/PI/code/working_on')
from stock_class import Stock
from portfolio_class import Portfolio
import matplotlib.pylab as plt
import numpy as np
import scipy.stats as st    
conn = sql.connect('F:/Nut/PI/data/data.db')
#%%


query = 'select distinct code from stocks_price where date = "2014-01-03"'  
code_list  = pd.read_sql(query, conn)

code_list = code_list.iloc[:,0]
code_list = code_list.tolist()


stock = Stock(conn, code_list,start='2014-01-03')
#%%
sys.path.append('F:/Nut/PI/code/working_on/Time_Series')
import Dev_DL.test_ts_dist as test_ts_dist
import importlib
importlib.reload(test_ts_dist)

from arma_garch import ARMA_GARCH


stock_ret = stock.daily_returns
stock_ret_series = stock_ret.values

n_stock = stock_ret.shape[1]
#
#arma_order = [2,0,2]
#garch_order = [1,1]
#
#model = ARMA_GARCH(order1=arma_order,order2=garch_order)

arma_order = [2,0,2]
model = ARMA_GARCH(order1=[1,0,1],order2=[1,1])

df_pass = np.zeros(100)
arma_param = []
garch_param = []
code_success = []
code_fail = []

for i in range(0,n_stock):
    # Test stationarity
    stock_ret_i = stock_ret.iloc[:,i]
    stock_ret_i = stock_ret_i.dropna()
    
    test_df = test_ts_dist.test_stationarity(stock_ret_i)
    if test_df[1]>0.1: df_pass[i] = 1
    # estimate ARMA
    try: 
        code_success.append(code_list[i])    
        model.estimation(stock_ret_i)
        arma_param.append(model.arma_params)
        garch_param.append(model.garch_params)
    except:
        code_fail.append(code_list[i])    




for i in range(0,len(code_fail)):
    print(code_fail[i]+' min = %.4f '%np.min(stock_ret[code_fail[i]].dropna()))
for i in range(0,len(code_fail)):
    print(code_fail[i]+' max = %.4f '%np.max(stock_ret[code_fail[i]].dropna()))
    
for i in range(0,len(code_fail)):  
    print('%.4f'%np.min(stock_ret[code_fail[i]].dropna()))
    
for i in range(0,len(code_fail)):  
    print('%.4f'%np.max(stock_ret[code_fail[i]].dropna()))
#    plt.figure() 
#    plt.plot(stock_ret[code_fail[i]].dropna())
