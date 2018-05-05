#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:05:20 2018
@author: zifandeng
"""

import sqlite3 as sql
import Ratios as ratio
from stock_class import Stock
import general_tools as tool
import pandas as pd
import numpy as np
import functools
import time
import statsmodels.api as sm
import arch as arch
from Time_Series.arma_garch import ARMA_GARCH
import warnings

#%%

def get_date_range(conn,t,window,frequency,method = 'backward'):
    bus_calender = tool.get_business_calendar(conn)
    if(bus_calender[bus_calender['date']==t].empty):
        raise ValueError('date '+t+' is not a business day!')
    # check frequency
    if frequency == 'annually':
        date_range = rolling_bus_day(conn,window,t,pd.offsets.relativedelta(years=1),method)
    elif frequency == 'quarterly':
        date_range = rolling_bus_day(conn,window,t,pd.offsets.relativedelta(months=3),method)
    elif frequency == 'monthly':
        date_range = rolling_bus_day(conn,window,t,pd.offsets.relativedelta(months=1),method)
    elif frequency == 'weekly':
        date_range = rolling_bus_day(conn,window,t,pd.offsets.relativedelta(weeks=1),method)
    elif frequency == 'daily':
        date_range = rolling_bus_day(conn,window,t,pd.offsets.relativedelta(days=1),method)
    else:
        raise ValueError('Please enter a valid frequency!')
    return date_range

def get_mkt_index(conn,t,window,frequency,index_code):
    date_range = get_date_range(conn,t,window,frequency)
    begin = date_range[0]
    mkt = pd.read_sql("select date,close,code from index_price where date >='"+begin.strftime('%Y-%m-%d')+"' and date <='" +t+"'"+' and code="%s"' % (index_code),conn)
    mkt = mkt.sort_values(by = 'date')
    mkt['date'] = pd.to_datetime(mkt['date'])
    mkt = mkt.set_index('date')
    mkt['mkt_ret'] = np.log(mkt['close'])-np.log(mkt['close'].shift(1))
    mkt = mkt.dropna()
    for i in range(1,len(date_range)):
        mkt.loc[(mkt.index>date_range[i-1]) & (mkt.index<=date_range[i]),'group']=date_range[i]        
    mkt2 = pd.DataFrame(mkt.groupby('group')['mkt_ret'].sum())
    return mkt2

def get_coef(stk,mkt,riskfree,code):
    try:
        data = stk[[code]]
        data = data.dropna()
        data = pd.merge(data,mkt,'left',left_index= True,right_index = True)
        beta = ratio.get_beta(data[code],data['mkt_ret'],riskfree)
        alpha = ratio.get_annulized_alpha(asset=data[code],market=data['mkt_ret'],annualization=1,riskfree=riskfree)
        r2 = ratio.get_rsquare(asset=data[code],market=data['mkt_ret'],riskfree=riskfree)
        begin = data.index[0]
        end = data.index[~0]
        n= len(data[code])
    except:
        print('No Return Data Available for '+ code)
        beta = np.nan
        alpha = np.nan
        r2 = np.nan
        begin = np.nan
        end = np.nan 
        n = 0
    return code,begin,end,n,alpha,beta,r2


def rolling_bus_day(conn,window,t,d_offsets,method):
    bus_calender = tool.get_business_calendar(conn)
    if(bus_calender[bus_calender['date']==t].empty):
        raise ValueError('date '+t+' is not a business day!')
    date_range=[]
    date_temp3 = pd.to_datetime(t)
    date_range.append(t)
    for i in range(1,window):
        if method == 'backward':
            date_temp = (date_temp3-d_offsets*i)
        else:
            date_temp = (date_temp3+d_offsets*i)
        #date_temp2 =  bus_calender[bus_calender<=date_temp.strftime('%Y-%m-%d')].dropna().tail(1)   
        date_temp2 = date_temp.strftime('%Y-%m-%d')
        date_range.append(date_temp2)
        #date_temp3= pd.to_datetime(date_temp2.iloc[0,0])
    date_range = pd.to_datetime(date_range).sort_values()
    return date_range

def capm_modeling(conn,t,frequency,window,stk_list = None,riskfree=0.0,mkt_code = 'sh000300',
                  stocks_price_old=None, business_calendar=None, industry=None):
    
    """
    Stand at time t,
    Run the CAPM model for a rolling period before time t (including t);
    
    Parameters
    ----------
    conn: 
        sqlite3 connection;
    t: 
        string; a string of time in form of 'yyyy-mm-dd';
    frequency:
        string; a string of frequency for returns 
                i.e., 'daily','weekly','monthly','quarterly','annually';
    code_list:
        list; a list for stock code, default None;
              if None, then the entire pool of stock as of time t would be considered
    
    riskfree:
        float; a floating number for risk free rate, default 0.0
    
    window:
        int; an integer for rolling window input, in the unite of frequency
             i.e., if window = 2, frequency = 'weekly', it means 2 weeks
    mkt_code:
        string; a string for market index, default 'sh000300'

    Returns
    -------
    coeficient matrices: pd.DataFrame
        the matrix containing the information of 
        [code, available beginning date, end date, number of observation, alpha, beta, r_squared] 
    """  
    # Get Business Calender to check if selected date is business day
    since = time.time()
    bus_calender = tool.get_business_calendar(conn)
    if(bus_calender[bus_calender['date']==t].empty):
        raise ValueError('date '+t+' is not a business day!')
    date_range = get_date_range(conn,t,window,frequency)
    begin = date_range[0]
    if stk_list == None:
        stk_list = tool.get_stock_pool(conn,t)
        data = Stock(conn,[],begin.strftime('%Y-%m-%d'),t,all_stocks=True,
                     stocks_price_old=stocks_price_old,
                     business_calendar=business_calendar,
                     industry=industry).daily_returns
        data = data[stk_list['code']]
    else:
        stk_list = pd.DataFrame(stk_list)
        stk_list.columns = ['code']
        data = Stock(conn,list(stk_list['code'][0:500]),begin.strftime('%Y-%m-%d'),t,
                     stocks_price_old=stocks_price_old,
                     business_calendar=business_calendar,
                     industry=industry).daily_returns
        index = 500
        while index < (stk_list.index[~0]+1):
            index_end = np.minimum(index+500,(stk_list.index[~0]+1))
            data = pd.merge(data,Stock(conn,list(stk_list['code'][index:index_end]),begin.strftime('%Y-%m-%d'),t).daily_returns,how = 'outer',left_index = True,right_index =True)
            index=index_end
    # get the time period before and including t
    mkt = get_mkt_index(conn,t,window,frequency,mkt_code)[['mkt_ret']] 
    # Extract all stocks as of time t
    for i in range(1,len(date_range)):
        data.loc[(data.index>date_range[i-1]) & (data.index<=date_range[i]),'group']=date_range[i]             
    data2 = pd.DataFrame(data.groupby('group').agg(lambda x:x.sum(skipna = False)))
    # Calculate coeficients    
    results = (map(functools.partial(get_coef,data2,mkt,riskfree),stk_list['code']))
    model_coef = list(results)
    time_lag = time.time()-since        
    cols = ['code','begin','end','number of obs','alpha','beta','r2']
    model_coef = pd.DataFrame(model_coef,columns = cols)
    print('Coefficient calculation completed in {:.0f}m {:.0f}s'.format(time_lag // 60, time_lag % 60))
    
    industry_query = 'select code,industry from stock_basics'
    industry = pd.read_sql(industry_query,conn)
    model_coef = pd.merge(model_coef,industry,left_on = 'code',right_on = 'code',how = 'left')
    # return
    model_coef_sub = model_coef[model_coef['number of obs']>50]
    industry_avg_beta = pd.DataFrame(model_coef_sub.groupby('industry')['beta'].apply(np.nanmean))
    industry_avg_beta.columns = ['industry_avg_beta']
    industry_avg_alpha = pd.DataFrame(model_coef_sub.groupby('industry')['alpha'].apply(np.nanmean))
    industry_avg_alpha.columns = ['industry_avg_alpha']
    model_coef = pd.merge(model_coef,industry_avg_beta,left_on = 'industry',right_index = True,how = 'left')
    model_coef = pd.merge(model_coef,industry_avg_alpha,left_on = 'industry',right_index = True,how = 'left')
    return model_coef

def mkt_forecasting(conn,t,frequency,lookback_window,proj_period,method,mkt_code = 'sh000300',n_periods=15,annualization = 252,arma_order=[2,0,0],garch_order=[1,1]):
    # Stand at time t 
    # Predict t+n
    bus_calender = tool.get_business_calendar(conn)
    if(bus_calender[bus_calender['date']==t].empty):
        raise ValueError('date '+t+' is not a business day!')
    mkt = get_mkt_index(conn,t,lookback_window,frequency,mkt_code)[['mkt_ret']]
    if method == 'hist_mean':
        mkt_pre = np.repeat(np.mean(mkt),proj_period)
    elif method == 'n_period_avg':
        mkt_pre = np.repeat(np.mean(mkt.tail(n_periods)),proj_period)
    elif method == 'arma_garch':
        # Test AR effect
        ar_effect = tool.autoregression_test(mkt['mkt_ret'])
        # If Yes, automatically detect an order with minimun AIC as best order
        # Fit this ARMA model and get residual as garch modelling data
        if ar_effect == True:
            arma_order = sm.tsa.stattools.arma_order_select_ic(mkt['mkt_ret'],ic='aic')['aic_min_order']
            arma_order = [arma_order[0],0,arma_order[1]]
            model_arma = sm.tsa.ARIMA(mkt['mkt_ret'],order = arma_order).fit()
            garch_data = model_arma.resid
        # If no, using historical mean as mu, get residual as garch data 
        else:
            garch_data = mkt['mkt_ret']-np.mean(mkt['mkt_ret'])
        # Test garch effect
        garch_effect = tool.garch_effect_test(garch_data)
        # If Garch and AR effect both exist, using arma garch with best order
        # If only Garch effect exists, apply only garch model on returns
        if (garch_effect == True):
            if(ar_effect == True):
                #model = ARMA_GARCH(arma_order,garch_order)
                #model.estimation(mkt)
                #model.prediction(proj_period)
                #mkt_pre = model.prediction_x
                mkt_pre = model_arma.forecast(proj_period)[0]
            else:
                warnings.warn('No autoregression effect detected by Ljung Box test. Use GARCH only!')
                model =  arch.arch_model(mkt['mkt_ret'],p=1,q=1).fit()
                mkt_pre = np.array(model.forecast(horizon = proj_period).mean.tail(1))
        # If neither exists, using historical mean instead
        elif (ar_effect == True):
            mkt_pre = model_arma.forecast(proj_period)[0] 
        else:
            warnings.warn('No autoregression and garch effect detected by Ljung Box test, use historical mean instead!')
            mkt_pre = np.repeat(np.mean(mkt),proj_period)                      
    elif method == 'arma':
        # detect ar effect
        ar_effect = tool.autoregression_test(mkt['mkt_ret'])
        # if true, automatically select best order and predict
        if ar_effect == True:
            arma_order = sm.tsa.stattools.arma_order_select_ic(mkt['mkt_ret'],ic='aic')['aic_min_order']
            arma_order = [arma_order[0],0,arma_order[1]]
            model = sm.tsa.ARIMA(mkt['mkt_ret'],order = arma_order).fit()
            mkt_pre = model.forecast(proj_period)[0]
        # if false, use historical mean
        else:
            warnings.warn('No autoregression detected by Ljung Box test, use historical mean instead!')
            mkt_pre = np.repeat(np.mean(mkt),proj_period)
    else:
        raise ValueError('Please enter a valid method: hist_mean/n_period_avg/arma_garch/arma')
    
    proj_date = get_date_range(conn,t,proj_period+1,frequency,'forward')
    proj_date = proj_date[range(1,len(proj_date))]
    mkt_pre = pd.DataFrame(np.array(mkt_pre),index=proj_date)
    return (mkt_pre)

def coef_multiplication(coef,mkt_pre):
    coef = coef.reset_index(drop=True)
    if coef['number of obs'][0]<=50:
        return coef['industry_avg_alpha']+coef['industry_avg_beta']*(mkt_pre)
    else:
        return coef['alpha']+coef['beta']*(mkt_pre)

def stock_forecasting(coef,mkt_pre,riskfree=0.0):
    mkt_pre = mkt_pre - riskfree
    stk_ret = coef.groupby('code')[['alpha','beta','number of obs','industry_avg_alpha','industry_avg_beta']].apply(coef_multiplication,mkt_pre).reset_index()
    stk_ret = stk_ret.dropna()
    stk_ret.columns = ['code','date','proj_ret']
    stk_ret['proj_ret'] = stk_ret['proj_ret']+riskfree
    # Highlighted small data set
    return stk_ret

def capm_mu(conn, start, lookback_win, 
            stk_list = None, proj_period=4, proj_method='arma_garch', freq='weekly',
            arma_order = [2, 0, 0], garch_order=[1, 1],
            stocks_price_old=None, business_calendar=None, industry=None, **kwargs):
    # step 1 get capm coeficients
    model_coef = capm_modeling(conn,start,freq, lookback_win, stk_list=stk_list,
                               stocks_price_old=stocks_price_old,
                               business_calendar=business_calendar,
                               industry=industry)
    # step 2 get market prediction
    mkt_prediction = mkt_forecasting(conn, start, freq, lookback_win, proj_period, proj_method,
                                     arma_order=arma_order, garch_order=garch_order)
    # step 3 get stock return forecasts
    stk_prediction= stock_forecasting(model_coef,mkt_prediction)
    # annualization
    freq_n = {'annually':252,
              'quarterly':63,
              'monthly': 21,
              'weekly':5,
              'daily':1}
    stk_prediction = stk_prediction.groupby('code')['proj_ret'].mean()/freq_n[freq]*252
    return stk_prediction