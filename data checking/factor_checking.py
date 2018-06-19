#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:28:42 2018

@author: yunongwu
"""

import pandas as pd
import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import sys
conn = '/Users/yunongwu/Nustore Files/PI/Code/Working_On/Factor_Model/modeling_data_wind_2001-01-01_2017-12-31.pkl'
model_data = pd.read_pickle(conn)
#%% example

macd1=model_data[model_data['code']=='000553'][['code','date','MACD']]
macd1=macd1.set_index('date')
mt.plot(macd1.index,macd1[['MACD']])

#%%

def factor_statistics(model_data,factor_name):
    modeldata=model_data[['code','date',factor_name]]
    mean=modeldata.groupby('code')[[factor_name]].apply(np.mean)
    mean=pd.DataFrame({'Mean':mean[factor_name]})
    mean=mean.reset_index()
    maximum=modeldata.groupby('code')[[factor_name]].apply(np.max)
    maximum=pd.DataFrame({'Max':maximum[factor_name]})
    maximum=maximum.reset_index()
    minimum=modeldata.groupby('code')[[factor_name]].apply(np.min)
    minimum = pd.DataFrame({'Min':minimum[factor_name]})
    minimum = minimum.reset_index()
    dfs = [mean,maximum,minimum]
    df_final = reduce(lambda left,right: pd.merge(left,right,on='code'), dfs)
    return df_final

#macd=factor_statistics(model_data,'MACD')
#liquidity=factor_statistics(model_data,'liquidity')
#size=factor_statistics(model_data,'size')
#leverage=factor_statistics(model_data,'leverage')
#Beta=factor_statistics(model_data,'Beta')
#volatility=factor_statistics(model_data,'volatility')

factor_list = ['liquidity','size','leverage','Beta','volatility','MACD']

for i , row in enumerate(factor_list):
    dfName = factor_list[i]
    factor_summary=factor_statistics(model_data,dfName)
    globals()[dfName]=factor_summary

#%%
    

        
hs300=pd.read_excel('/Users/yunongwu/Desktop/hs300.xlsx')
hs300.code = hs300.code.str[:6]

MACD300=pd.merge(hs300,MACD)
liquidity300=pd.merge(hs300,liquidity)
leverage300=pd.merge(hs300,leverage)
Beta300=pd.merge(hs300,Beta)
volatility300=pd.merge(hs300,volatility)
size300=pd.merge(hs300,size)

hs279=MACD300.code

def factor_plot(factor_name,code):
    modeldata=model_data[model_data['code']==code][['code','date',factor_name]]
    modeldata=modeldata.set_index('date')
    plt.figure()
    plt.plot(modeldata.index,modeldata[[factor_name]])
    plt.title(factor_name+code)
    plt.savefig('/Users/yunongwu/Nustore Files/PI/Code/Working_On/data checking/Factor Checking/'+ factor_name+'/'+code+'.png')

for code in hs279:
    for factor in factor_list:
        factor_plot(factor,code)