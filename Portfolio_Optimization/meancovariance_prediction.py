#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:40:36 2018

@author: yunongwu
"""

import pandas as pd
import numpy as np
from stock_class import Stock
from arma_garch import ARMA_GARCH




def arma_garch_prediction(conn,code_list,start1,start2,end2,
                          arma_order = [1,0,1],garch_order = [1,1]):
    
    stock = Stock(conn, code_list,start=start2,end=end2)
    ts_data = stock.daily_returns
    correlation = ts_data[-250:].corr()
    ts_data = ts_data[start1:]
    
    model = ARMA_GARCH(order1=arma_order,order2=garch_order)
    
    for i in range(len(code_list)):
        return_data=ts_data.iloc[:,[i]].dropna()
        try:
            model.estimation(return_data)
            model.prediction(30)
        except ValueError:
            continue
        if i == 0: 
            pre_return_new=pd.DataFrame({ts_data.columns[i]:model.prediction_x})
            pre_vol_new=pd.DataFrame({ts_data.columns[i]:model.prediction_conditional_vol})
        else:
            pre_return=pd.DataFrame({ts_data.columns[i]:model.prediction_x})
            pre_return_new=pd.concat([pre_return_new,pre_return],axis=1)
            pre_return_avg=pre_return_new.mean()
            pre_vol=pd.DataFrame({ts_data.columns[i]:model.prediction_conditional_vol})
            pre_vol_new=pd.concat([pre_vol_new,pre_vol],axis=1)
            pre_vol_avg=pre_vol_new.mean()
        
    list1=list(pre_return_new.columns)
    list2=list(ts_data.columns)
    remaining_list = np.setdiff1d(list2,list1)
    ts_data_remain=ts_data[remaining_list]
    remain_mean=ts_data_remain.mean()
    remain_std=ts_data_remain.std()
    ereturn=pre_return_avg.append(remain_mean)
    evol=pre_vol_avg.append(remain_std)
    new_order=np.array(correlation.columns)
    evol=evol.reindex(new_order)
    ereturn=ereturn.reindex(new_order)
    
    corr=np.asmatrix(correlation)
    evol=np.asmatrix(evol)
    cov=pd.DataFrame(np.multiply((evol.T*evol), corr))
    
    return ereturn*252,cov*252,ts_data_remain