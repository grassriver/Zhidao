# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:08:40 2018

@author: Hillary

Generate covariance matrices for optimazation based on 
given codelist and date

"""

import pandas as pd
import numpy as np
#import datetime as dt
#import random
import sqlite3 as sql
#from pandas.tseries.offsets import BDay

#import statsmodels.tsa as ts
#import statsmodels.api as sm
#import functools
#import time
from numpy.linalg import LinAlgError

import sys
import os

path = os.path.abspath('./')
if path.split('\\')[-1] == 'Working_On':
    sys.path.append(path)
    dbpath = os.path.abspath('../../')+'\\Data\\data.db'
    conn = sql.connect(dbpath)
elif path.split('/')[-1] == 'Working_On':
    sys.path.append(path)
    dbpath = os.path.abspath('../../')+'/Data/data.db'
    conn = sql.connect(dbpath)
else:
    raise ValueError('enter Working_On path!')

#import sklearn.metrics as metrices 
#import sklearn.linear_model as lm
#import sys
#sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On')
#sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model')
#sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Time_Series')

from Time_Series.arma_garch import ARMA_GARCH
#import general_tools as tool
import Time_Series.test_ts_dist as test_ts_dist

#from imp import reload


#conn=sql.connect('C:/Users/Hillary/Documents/PI/data/data.db')


#conn1 = 'C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/modeling_data_2001-01-01_2017-12-31.pkl'


#ran_list =random.sample(range(200,2000),10) 
#code_list =model_data.iloc[ran_list]['code'].drop_duplicates().tolist()



#%%########################## load input data #################################


    
def generate_feed(code_list, anchor_date, window, model_data, wls_resid, conn):
        
    
    model_data_in = model_data.loc[(model_data['code'].isin(code_list))]
    
    if len(model_data_in.code.unique()) < len(code_list):
        
        raise ValueError('factor value not available for %s'%list(set(code_list) - set(model_data.code.unique())))
        

    wls_resid_used = wls_resid[(wls_resid['date']<anchor_date)&
                               (wls_resid['code'].isin(code_list))]
    
    if len(wls_resid_used.code.unique()) < len(code_list):
       
        raise ValueError('historical residual variance not available for %s'%list(set(code_list) -  set(wls_resid_used.code.unique())))
        
    
    wls_beta = pd.read_pickle(beta_path(conn))
    
    wls_beta = wls_beta.sort_values(by='date')
   
    
    factor_feed = model_data_in[model_data_in['date']<anchor_date].sort_values(by=['code','date'])
                            
    factor_feed = factor_feed.groupby(['code'],as_index=False).last()
    
    wls_beta_used = wls_beta[wls_beta['date']<anchor_date].tail(window)
    #.set_index('date',drop=False)
        
    wls_resid_used = wls_resid_used.sort_values(by=['code','date']).groupby('code',as_index=False).tail(window)
        
    return factor_feed, wls_beta_used, wls_resid_used

#[factor_feed,wls_beta_used,wls_resid_used]=generate_feed(code_list, anchor_date, 500)


def beta_ARMA_GARCH(df,factor_list,f_period=1):
    
    beta_forecast = pd.DataFrame()
    beta_conditional_vol = pd.DataFrame()
    
    for ff in factor_list:
          
          try:
                model = ARMA_GARCH(order1=[2,0,2],order2=[1,1])
                
                model.estimation(df[ff].values)
                model.prediction(f_period)
        
                beta_forecast[ff]=model.prediction_x
                beta_conditional_vol[ff] = model.prediction_conditional_vol
                
                print('ARMA-Garch modeling for factor %s: '%ff)
             
          except (ValueError, LinAlgError):
              
                model = ARMA_GARCH(order1=[0,0,2],order2=[1,1])
              
                model.estimation(df[ff].values)
                model.prediction(f_period)
        
                beta_forecast[ff]=model.prediction_x
                beta_conditional_vol[ff] = model.prediction_conditional_vol
                
                print('MA-Garch modeling for factor %s: '%ff)
                
                
                
    beta_forecast['f_period']=np.arange(1,f_period+1)      
    #beta_conditional_vol['f_period']=np.arange(1,f_period+1)           
          
    return beta_forecast,   beta_conditional_vol      

#[PreBeta,PreBetaVol]=beta_ARMA_GARCH(wls_beta_used,factor_list2,f_period=1)

def beta_hist_corr(wls_beta_used,factor_list):
        
    wls_beta_demean = wls_beta_used.copy()

    wls_beta_demean[factor_list] = wls_beta_demean[factor_list].apply(lambda x: x-x.mean())

    HistBetaCov =pd.DataFrame((np.matrix(wls_beta_demean[factor_list]).T*
                              np.matrix(wls_beta_demean[factor_list]))/(wls_beta_demean.shape[0]-1),
                              columns =factor_list)
    
    HistBetaVar = pd.DataFrame({'HistBetaVar':np.diagonal(np.matrix(HistBetaCov))})

    HistBetaVar['HistBetaStd']=HistBetaVar['HistBetaVar'].apply(np.sqrt)

    HistBetaVar['beta'] = pd.Series(factor_list)

    HistBetaStdMul = np.matrix(HistBetaVar.HistBetaStd).T*np.matrix(HistBetaVar.HistBetaStd)

    HistBetaCorr = pd.DataFrame(np.divide(np.matrix(HistBetaCov),HistBetaStdMul),
                            columns=factor_list)

    HistBetaCorr['beta'] = pd.Series(factor_list)
    
    return HistBetaCorr


def forecast_beta_cov(PreBetaVol, HistBetaCorr, factor_list,f_day=1):
    
    if f_day>len(PreBetaVol):
        raise ValueError('!!!chosen forecast period is longer than the series of beta forecasts') 

    row = f_day -1 
    
    PreBetaStdMul = np.matrix(PreBetaVol.iloc[row,:]).T*np.matrix(PreBetaVol.iloc[row,:])

    PreBetaCov = pd.DataFrame(np.multiply(np.matrix(HistBetaCorr[factor_list]),PreBetaStdMul),
                          columns = factor_list)
    
    return PreBetaCov

#PreBetaCov = forecast_beta_cov(PreBetaVol, HistBetaCorr, factor_list2, f_day=1)

def forecast_ret_cov1(feed, BetaCov,code_list,factor_list):
    
    PreRetCov = pd.DataFrame(np.matrix(feed[factor_list]) 
                         *np.matrix(BetaCov[factor_list]) 
                         *np.matrix(feed[factor_list]).T ,columns =code_list )

    PreRetCov['code'] = pd.DataFrame(code_list)
    
    return PreRetCov

#PreRetCov = forecast_ret_cov1(factor_feed, PreBetaCov,code_list,factor_list2)

def forecast_ret_cov2(code_list, feed,wls_beta_used, wls_resid_used, factor_list,f_day=1):
    
    '''
    f_day: the ith day of forecast
    feed: factor values used as forecast input
    PreBetaVol: beta variance forecasts from ARMA-GARCH module
    HistBetaCorr: historical beta correlation matrix
    
    '''
    [PreBeta,PreBetaVol] = beta_ARMA_GARCH(wls_beta_used, factor_list,f_period=1)     
    
    HistBetaCorr = beta_hist_corr(wls_beta_used,factor_list)

    PreBetaCov = forecast_beta_cov(PreBetaVol, HistBetaCorr, factor_list,f_day=1)

    PreRetCov = forecast_ret_cov1(feed, PreBetaCov,code_list,factor_list)

    #start_date = anchor_date - BDay(252)
    #start_date = wls_resid.date.max() - dt.timedelta(days=365)

    #resid_var_in =wls_resid.loc[(wls_resid['code'].isin(code_list))&(wls_resid['date']>=start_date)]
    
    resid_var=pd.DataFrame({'resid_var':wls_resid_used.groupby('code')['wls_resid2'].apply(lambda x: x.sum()/(x.count()-1))})

    resid_var = resid_var.reset_index()    
    
    PreRetCov = pd.merge(PreRetCov, resid_var, on='code',how='left')
    
    PreRetCov['resid_var']=PreRetCov['resid_var'].transform(lambda x:x.fillna(x.median()) )
    
    for i in range(len(PreRetCov)):
        PreRetCov.iloc[i,i] = PreRetCov.iloc[i,i]+PreRetCov.loc[i,'resid_var']
    
    PreRetCov = PreRetCov.drop('resid_var',axis=1) 
   
    return PreRetCov

#PreRetCov = forecast_ret_cov2(factor_feed,wls_beta_used, wls_resid_used, factor_list2,f_day=1)

def time_series_stationarity(df,factor_list):
    
    df = df.sort_values(by='date')
    #df = df.reset_index()
    
    for ff in factor_list:
       
       print ('Dickey-Fuller test for factor %s'%ff)
       test_df = test_ts_dist.test_stationarity(df[ff])
       
       print('Ljung-Box test on autocorrelation for factor %s'%ff)
       test_ljq = test_ts_dist.ljungbox_test(df[ff]) 

#%%############
def factor_path(conn):
    
    return conn+ '/modeling_data_wind_2001-01-01_2017-12-31.pkl'
    
def beta_path(conn):
    
    return conn+ '/Country&Beta_WLS_Beta.pkl'
    
def resid_path(conn):
    
    return conn+ '/Country&Beta_WLS_resid_var.pkl'

def load_data(conn):

    model_data = pd.read_pickle(factor_path(conn))
    
    wls_resid= pd.read_pickle(resid_path(conn))
    
    wls_resid.code = wls_resid.code.astype(str)
    
    ind_list = (model_data.columns[pd.Series(model_data.columns).str.startswith('ind_')]).tolist()
    
    factor_list2 = ['liquidity','size','leverage','Beta','country',
                     'volatility','MACD',]+list(ind_list)

    
    keep = ['code','year','date','log_ret']+factor_list2
           
    model_data = model_data[keep]       

    
    return model_data, wls_resid, factor_list2



    
       
def barra_stk_cov(code_list, start, window, model_data, wls_resid, factor_list2, conn, **kwargs):

    # generate data used for modeling
    [factor_feed,wls_beta_used,wls_resid_used]=generate_feed(code_list, start, 
                                            window, model_data, wls_resid, conn)
    
#    time_series_stationarity(wls_beta_used, factor_list2)  

    PreRetCov = forecast_ret_cov2(code_list, factor_feed,wls_beta_used, 
                                  wls_resid_used, factor_list2,f_day=1)
    PreRetCov = PreRetCov.iloc[:,0:-1]

    return PreRetCov

def generate_feed_all(anchor_date, window, model_data, wls_resid, conn):
        
    code_list = list(set(wls_resid['code'].unique()) &
                     set(model_data.loc[model_data.date==anchor_date, 'code'].unique()))
    
    model_data_in = model_data.loc[(model_data['code'].isin(code_list))]
    
#    if len(model_data_in.code.unique()) < len(code_list):
#        
#        raise ValueError('factor value not available for %s'%list(set(code_list) - set(model_data.code.unique())))


    wls_resid_used = wls_resid[(wls_resid['date']<anchor_date)&
                               (wls_resid['code'].isin(code_list))]
    
#    if len(wls_resid_used.code.unique()) < len(code_list):
#       
#        raise ValueError('historical residual variance not available for %s'%list(set(code_list) -  set(wls_resid_used.code.unique())))
        
    
    wls_beta = pd.read_pickle(beta_path(conn))
    
    wls_beta = wls_beta.sort_values(by='date')
   
    
    factor_feed = model_data_in[model_data_in['date']<anchor_date].sort_values(by=['code','date'])
                            
    factor_feed = factor_feed.groupby(['code'],as_index=False).last()
    
    wls_beta_used = wls_beta[wls_beta['date']<anchor_date].tail(window)
    #.set_index('date',drop=False)
        
    wls_resid_used = wls_resid_used.sort_values(by=['code','date']).groupby('code',as_index=False).tail(window)
        
    return factor_feed, wls_beta_used, wls_resid_used, code_list


def barra_stk_cov_all(start, window, model_data, wls_resid, factor_list2, conn, **kwargs):

    # generate data used for modeling
    [factor_feed,wls_beta_used,wls_resid_used, code_list]=generate_feed_all(start, 
                                            window, model_data, wls_resid, conn)

    PreRetCov = forecast_ret_cov2(code_list, factor_feed,wls_beta_used, 
                                  wls_resid_used, factor_list2,f_day=1)
    PreRetCov = PreRetCov.iloc[:,0:-1]

    return PreRetCov

#def barra_stk_cov_all(start, window, model_data, wls_resid, factor_list2, conn, **kwargs):
#
#    # generate data used for modeling
#    [factor_feed,wls_beta_used,wls_resid_used]=generate_feed_all(start, 
#                                            window, model_data, wls_resid, conn)
#    
##    time_series_stationarity(wls_beta_used, factor_list2)  
#    
#    code_list = list(set(factor_feed['code'].unique()) &
#                     set(wls_resid_used['code'].unique()))
#    
#    factor_feed = factor_feed[np.in1d(factor_feed.code, code_list)]
#    wls_resid_used = wls_resid_used[np.in1d(wls_resid_used.code, code_list)]
#
#    PreRetCov = forecast_ret_cov2(code_list, factor_feed,wls_beta_used, 
#                                  wls_resid_used, factor_list2,f_day=1)
#
#    return PreRetCov

def barra_reverse_mu_all(start, window, model_data, wls_resid, factor_list2, data_path, dbpath, **kwargs):
    
    sigma = barra_stk_cov_all(start, window, model_data, wls_resid, factor_list2, data_path)
    code_list_all = sigma.code.tolist()
    sigma = sigma.set_index('code').sort_index()
#    sigma_all = np.matrix(sigma)
    
    conn = sql.connect(dbpath)
    query1 = 'select code, close from stocks_price where date = "{}"'.format(start)
    stocks = pd.read_sql(query1, conn)
    stocks = stocks.loc[np.in1d(stocks['code'], code_list_all), :].sort_values('code')
    
    query2 = 'select code, outstanding from stock_basics'
    outstanding = pd.read_sql(query2, conn).sort_values('code')
    
    stocks = pd.merge(stocks, outstanding, on='code', how='left')
    stocks['cap'] = stocks['close']*stocks['outstanding']
    stocks['weight'] = stocks['outstanding']/(stocks['outstanding'].sum())
    
    code_list_all = list(set(code_list_all) & set(stocks.code))
    
    sigma_all = sigma.loc[np.in1d(sigma.index, code_list_all), np.in1d(sigma.columns, code_list_all)]
    sigma_all = np.matrix(sigma_all)
    
    w = np.mat(stocks.weight).T
    delta = 0.5/np.sqrt(w.T*sigma_all*w)[0,0]
    mu = delta*sigma_all*w
    mu_all = pd.DataFrame({'code':code_list_all, 'mu':mu.reshape((len(mu),)).tolist()[0]})
    
    return mu_all

def barra_reverse_mu(code_list, start, window, model_data, wls_resid, factor_list2, data_path, dbpath, **kwargs):
    
    mu_all = barra_reverse_mu_all(start, window, model_data, wls_resid, factor_list2, data_path, dbpath)
    
    mu = mu_all.loc[np.in1d(mu_all.code, code_list), 'mu']
    return mu

#%%
if __name__ == '__main__':
    conn = path + '\\Factor model'
    anchor_date = '2017-01-03'
    code_list = ['600825','600229','600318','600743','600795','600794']
    [model_data, wls_resid, factor_list2]= load_data(conn)
    PreRetCov = barra_stk_cov(code_list,'2017-01-03',500)
