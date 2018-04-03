
# coding: utf-8

# In[184]:


# -*- coding: utf-8 -*-
"""
Created on 2018-03-13 19:14:30

@author: Daniel Lu
"""

import pandas as pd
import sqlite3 as sql
import sys
sys.path.append('/Users/tianyulu/Nustore Files/PI//code/working_on')
from stock_class import Stock
from portfolio_class import Portfolio
import matplotlib.pylab as plt
import numpy as np
import scipy.stats as st    
conn = sql.connect('/Users/tianyulu/Nustore Files/PI/Staff Working File/D.Lu/data.db')


code_list = ['000001']
stock = Stock(conn, code_list,start='2017-01-03')
# select one stock returns
ts_data = stock.daily_returns['000001']


# In[186]:


sys.path.append('/Users/tianyulu/Nustore Files/PI//code/working_on/Time Series')
import test_ts_dist 

# Dickey-Fuller test on stationarity
test_df = test_ts_dist.test_stationarity(ts_data)

# Linjung-Box test on autocorrelation

test_ljq = test_ts_dist.ljungbox_test(ts_data) 

# test distribution
test_dist = test_ts_dist.best_fit_distribution(ts_data)





#ACF and PACF plots:

test_acf = test_ts_dist.acf_pacf(ts_data)
test_pacf = test_ts_dist.acf_pacf(ts_data,'pacf')



# In[329]:


# Fit ARMA (2,2) and GARCH (1,1) model 

arma_order = [2,0,2]
garch_order = [1,1]

model = ARMA_GARCH(order1=arma_order,order2=garch_order)


model.estimation(ts_data)

#%%
# predict next n period based on fitted ARMA-GARCH model
model.prediction(10)


#%%

# KS test

innovation = model.arma_fitted/model.garch_fitted
innovation_norm = (innovation-np.mean(innovation))/np.std(innovation)

D, p_value = st.stats.kstest(innovation_norm,'norm',alternative='two-sided')
print ('reject' if p_value<0.05 else 'not reject')


