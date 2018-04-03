#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:04:03 2018

@author: zifandeng
"""

import pandas as pd
import sqlite3 as sql
import sys
import numpy as np
sys.path.append('/Users/zifandeng/Nustore Files/PI/Code/Working_On')
import general_tools as tool

#%%
conn=sql.connect('/Users/zifandeng/Nustore Files/PI/data/data.db')

#%%
# get stock data
def get_data(conn,start,end):
    stock_prices = pd.read_sql("select a.*,b.industry,b.area,b.outstanding,b.timetomarket from stocks_price a left join stock_basics b on a.code = b.code where date >='"+start+"' and date <= '"+end+"' order by a.date,a.code asc",conn)
    # date transfer
    stock_prices['quarter'] = stock_prices.date.apply(tool.get_lagged_time,freq = 'quarterly',lag = 0)
    # get fundamental data
    min_quarter = stock_prices.quarter.min()
    max_quarter= stock_prices.quarter.max()
    fundamental = pd.read_sql("select distinct * from fundamental where date>='"+min_quarter+"' and date <='"+max_quarter+"'",conn)
    fundamental=fundamental.rename(columns={'date':'quarter'})
    stock_merged = stock_prices.merge(fundamental,right_on = ['code','quarter'],left_on=['code','quarter'],how='left')
    stock_merged['marketcap'] = stock_merged['close']*stock_merged['outstanding']
    return stock_merged
#%%
def check_outlier(s,distance,fill = False):
    s=pd.to_numeric(s,'coerce')
    avg = np.nanmean(s)
    std = np.nanstd(s)
    outlier = s[(s>avg+distance*std)|(s<avg-distance*std)]
    if (outlier.empty):
        with_outiler='N'
    else:
        with_outiler='Y'
    return with_outiler
    
def check_missing(s,fill = False):
    #missing = s[np.isnan(s)]
    if (s.hasnans):
        with_missing='Y'
    else:
        with_missing='N'
    return with_missing

def fill_data(s,distance):
    s = pd.to_numeric(s,'coerce')
    avg = np.nanmean(s)
    std = np.nanstd(s)
    s[(s>avg+distance*std)] = avg+distance*std
    s[(s<avg+distance*std)] = avg-distance*std
    s[np.isnan(s)]=np.nanmean(s)
    return s

def normalization(s):
    s = pd.to_numeric(s,'coerce')
    avg = np.nanmean(s)
    std = np.nanstd(s)
    s = (s-avg)/std
    return s