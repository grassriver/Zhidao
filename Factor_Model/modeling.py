#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:45:54 2018

@author: zifandeng
"""

"""
Updated by Hillary (Apr 8, 2018):
    (1) streamline modeling data input
    (2) allow flexibility of specifying different factor lists
    (3) construct function to generate beta forecasts based on specified forecast period 
        by feeding back beta forecast to for next period forecast
    (4) construct function to generate stock return,covariance, correlation forecasts based on beta forecasts
update (Apr 9, 2018):
    (1) test stationarity and autocorrelation of time-series of betas
    (2) construct function to call arma_garch module from time-series to fit ARMA-GARCH model to
    betas and residuals and generate forecasts for betas directly 
update (Apr 13, 2018):

"""    

'''   
########################################################
# Issue: beta for ind_13 is not stationary
########################################################
'''
 
import sqlite3 as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import statsmodels.tsa as ts
import statsmodels.api as sm

import sklearn.metrics as metrices 
import sklearn.linear_model as lm


#sys.path.append('/Users/zifandeng/Nustore Files/PI/Code/Working_On')

sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On')
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model')
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Time_Series')

import general_tools as tool
import test_ts_dist 
from arma_garch import ARMA_GARCH



#conn1 = 'C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/test_model_data.pkl'

conn1 = 'C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/modeling_data_2001-01-01_2016-12-31.pkl'


#%%####################### Pull Data and Specify Factor list #######################

model_data = pd.read_pickle(conn1)

ind_list = model_data.columns[pd.Series(model_data.columns).str.startswith('ind_')]


model_data.groupby('year').code.count()

# based on data stat, limited obs in 2001 compared to the rest of the periods
# and reserve one year for out-of-sample testing

insample = model_data[(model_data['date']>='2001-01-01')&(model_data['date']<='2016-12-30')]

sas_test = model_data[(model_data['date']>='2016-01-01')&(model_data['date']<='2016-12-30')]

keep=['log_ret','code','date','mkt_ret']+factor_list2

sas_test=sas_test[keep]     
     
     
     

sas_test.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/sas_test.csv')

outsample = model_data[(model_data['date']>='2017-01-01')&(model_data['date']<='2017-12-30')]

len(insample.industry_code.unique())


# specify factor list
varlist = ['MACD','GAP','liquidity','size','BM','leverage',
           'PE','roe_gr','eps_gr','volatility','marketcap',
           'cf_liabilities','cf_sales','roe',
           'mbrg','nav','net_profit_ratio','bvps']

factor_list1 = ['liquidity','size','BM','mkt_ret','leverage',
                'volatility','MACD',]+list(ind_list)

factor_list2 = ['liquidity','size','BM','leverage',
                'volatility','MACD',]+list(ind_list)

factor_list3 = list(ind_list)

#%%####################Cross-Section Regression to get time-series of Betas ###################

def build_model(df,mod,x_var,y_var):
    var_list = x_var.copy()
    var_list.insert(0,y_var)
    df = df[var_list]
    df = df.dropna()
    x = df[x_var]
    y = df[y_var]
    fit1 = mod.fit(x,y)
    y_hat = fit1.predict(x)
    return mod.coef_, mod.intercept_, fit1.score(x,y), metrices.mean_squared_error(y,y_hat)


def get_betas(data,x_var,y_var,x_lag = 1):
    beta =  pd.DataFrame()
    intercep = pd.DataFrame()
    r_2 =  pd.DataFrame()
    MSE =  pd.DataFrame()
    for t in data.date.unique():
        #t = t.strftime("%Y-%m-%d")
        #quarter = tool.get_lagged_time(t,'quarterly',x_lag)
        #x_var.insert(0,'code')
        #price = data.loc[data.date == t,['code',y_var]]
        #factors = data.loc[(data.year == int(quarter[0:4]))&(data.quarter == int(quarter[5:7])),x_var]
        #df = price.merge(factors,how ='left',left_on = 'code',right_on = 'code')
        print(t)
        [coef,intercept,r_square,mse] = build_model(data.loc[data.date == t,:],lm.LinearRegression(),x_var,y_var)
        beta=beta.append(pd.DataFrame([coef]))
        intercep=intercep.append(pd.DataFrame([intercept]))
        r_2=r_2.append(pd.DataFrame([r_square]))
        MSE=MSE.append(pd.DataFrame([mse]))
    
    beta = pd.DataFrame(beta)
    r_2 = pd.DataFrame(r_2)
    intercep = pd.DataFrame(intercep)
    MSE = pd.DataFrame(MSE)
    
    beta.columns = x_var    
    beta = beta.set_index(data.date.unique())
    r_2.columns = ['r2']
    r_2 = r_2.set_index(data.date.unique())
   
    return intercep,beta,r_2,MSE

[InSampleConstant,InSampleBeta,InSampleR2,InSampleMse]=get_betas(insample,factor_list2,'log_ret',x_lag=0)

#[InSampleConstant_fl1,InSampleBeta_fl1,InSampleR2_fl1,InSampleMse_fl1]=get_betas(insample,factor_list1,'log_ret',x_lag=0)


CSIntercept = InSampleConstant.mean()


InSampleR2[InSampleR2['r2']>0.3]

InSampleR2[InSampleR2_fl1['r2']<0.1].count()

InSampleR2.shape
YX = factor_list2 +['code','log_ret']
InSampleR2[InSampleR2['r2']==InSampleR2['r2'].min()]

lowest_r2 = insample[insample['date']=='2011-07-21'][YX]
lowest_r2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/lowest_r2.csv')



InSampleBeta['year']=InSampleBeta.index.year
InSampleBeta['quarter']=InSampleBeta.index.quarter

IB = InSampleBeta.reset_index()
            
InSampleBeta_Mean = IB.groupby(['year','quarter'],as_index =False).mean()   
InSampleBeta_Mean['yq']= InSampleBeta_Mean['year']*100+InSampleBeta_Mean['quarter']         

  

InSampleBeta_Mean.set_index('yq', inplace=True)       
plt.plot(InSampleBeta_Mean['size'],'k--')   
             
#%%########################## Modeling Time-Series of Betas ##############################

# test stationarity and autocorrelation of in-sample betas
def time_series_stationarity(beta,factor_list):
    
    # Dickey-Fuller test on stationarity
    for ff in factor_list:
       
       print ('Dickey-Fuller test for factor %s'%ff)
       test_df = test_ts_dist.test_stationarity(InSampleBeta[ff])
       
       print('Ljung-Box test on autocorrelation for factor %s'%ff)
       test_ljq = test_ts_dist.ljungbox_test(InSampleBeta[ff]) 
       
time_series_stationarity(InSampleBeta, factor_list2)   

time_series_stationarity(InSampleBeta_Mean, ['size'])   


time_series_stationarity(InSampleBeta,['ind_13','ind_14'])    


def time_series_analysis(beta,order):
    arcoef = pd.DataFrame()
    intercept = pd.DataFrame()
    
    for i in range(beta.shape[1]):
        
        tsmod = sm.tsa.ARMA(beta.iloc[:,i],order).fit()  
        coef = pd.DataFrame(tsmod.params[1:len(tsmod.params)]).reset_index(drop = True)
        intercep=pd.DataFrame([tsmod.params[0]]).reset_index(drop = True)
        
        if i == 0:          
            
            intercept=intercept.append(pd.DataFrame([tsmod.params[0]]))
            arcoef=arcoef.append(pd.DataFrame((tsmod.params[1:len(tsmod.params)])))
            arcoef.rename(columns = {arcoef.columns[i]:beta.columns[i]},inplace=True)  
            intercept.rename(columns = {intercept.columns[i]:beta.columns[i]},inplace=True)            
        
        else:
            
            intercept = pd.concat([intercept,intercep],1)
            arcoef=pd.concat([arcoef.reset_index(drop = True),coef],1)
            arcoef.rename(columns = {arcoef.columns[i]:beta.columns[i]},inplace=True)  
            intercept.rename(columns = {intercept.columns[i]:beta.columns[i]},inplace=True)             
    
    return arcoef,intercept

[arcoef,intercept]=time_series_analysis(InSampleBeta,(1,0))

InSampleBeta['date']=InSampleBeta.index
InSampleBeta = InSampleBeta.reset_index()    
        


#%%##################### Beta Forecasts #################################

'''
#Two approaches are available: (1) call time-series module and forecast multiple days at once
                               (2) call sm.tsa.ARMA and forecast beta by feeding eralier betas
                                   back to the AR model,without fitting GARCH model to residuals
                               
'''

"""
def ts_one_step_prediction(arcoef,intercept,realization):
    realization = realization.tail(arcoef.shape[0])
    realization=realization.reset_index(drop = True)
    beta = realization.rmul(arcoef).sum()+intercept
    return beta
"""

def ts_beta_prediction(arcoef,intercept,InSampleBeta,f_period):
    
    InSampleBeta = InSampleBeta.sort_values(by='date')
    
    ISBeta = InSampleBeta.tail(arcoef.shape[0])
    
    #beta_forecast=pd.DataFrame(columns=ISBeta.columns)
    #ISBeta=realization_used.copy()
    #beta_forecast = pd.DataFrame(index=range(f_period),columns=range(arcoef.shape[1]))
    
    beta_forecast = pd.DataFrame()
    
    for t in range(f_period):
        
        beta_f= ISBeta.sort_values(by='date').tail(arcoef.shape[0]).rmul(arcoef).sum()+intercept
        beta_f['date']=ISBeta['date'].max()+dt.timedelta(days=1)          
        
        #beta_f.index = ISBeta['date'].max().index+dt.timedelta(days=1)                              
        
        #beta_forecast = beta_forecast.append(beta_f)
        beta_forecast=beta_forecast.append(pd.DataFrame(beta_f))

        ISBeta = ISBeta.append(beta_f)

    return beta_forecast,  ISBeta

[PreBeta,ISBeta]=ts_beta_prediction(arcoef,intercept,InSampleBeta,5)

factor_list2=['liquidity','size','BM','mkt_ret',
                'volatility','MACD',]

def beta_ARMA_GARCH(InSampleBeta,model,factor_list,f_period):
    
    model = model
    beta_forecast = pd.DataFrame()
    beta_conditional_vol = pd.DataFrame()
    
    for ff in factor_list:
        
        print('modeling for factor beta %s: '%ff)
        model.estimation(InSampleBeta[ff].values)
        model.prediction(f_period)
        beta_forecast[ff]=model.prediction_x
        beta_conditional_vol[ff] = model.prediction_conditional_vol
                     
    return beta_forecast,   beta_conditional_vol      

[PreBeta,PreBetaVol] = beta_ARMA_GARCH(InSampleBeta, ARMA_GARCH(order1=[1,0,1],order2=[1,1]),factor_list1,5)     

[PreBeta,PreBetaVol] = beta_ARMA_GARCH(InSampleBeta, ARMA_GARCH(order1=[1,0,1],order2=[1,1]),factor_list2,8)     


#%%#################Stock Return, Covariance, Correlation Forecasts ######################


def stk_return_forecast(loading,PreBeta,factor_list,f_period):
    
    if f_period>len(PreBeta):
        raise ValueError('!!!chosen forecast period is longer than the series of beta forecasts') 
    
    load = loading[loading['date']==loading.date.max()]
    
    use = factor_list + ['code']
    
    load = load[use]
    load = load.set_index('code',drop=False)
    
    PreReturn = pd.DataFrame([])
    
    for t in range(f_period):
              
        ret = pd.DataFrame({'PreRet':load[factor_list].dot(PreBeta.iloc[t][factor_list])})+CSIntercept[0]
                
        ret=ret.reset_index()
        t+=1
        ret['f_period']=t
        
        PreReturn = PreReturn.append(ret)
           
        PreReturn['PreRet'] = PreReturn.groupby(by='f_period',as_index=False)['PreRet'].transform(lambda x: x.fillna(x.median()))
   
    PreReturn = PreReturn.drop_duplicates()
    
    pivot_ret =PreReturn.pivot_table(index='f_period',columns='code',values='PreRet') 
    
    PreRetCov = pivot_ret.cov()
    PreRetCorr = pivot_ret.corr()
    PreRetCov = PreRetCov.reset_index()
    PreRetCorr = PreRetCorr.reset_index()
    
    return PreReturn, PreRetCov, PreRetCorr

[PreReturn,PreRetCov,PreRetCorr]=stk_return_forecast(model_data,PreBeta,factor_list1,5)

PreReturn.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/test_FM_PreReturn.pkl")
PreRetCov.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/test_FM_PreRetCov.pkl")
PreRetCorr.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/test_FM_PreRetCorr.pkl")


[PreReturn,PreRetCov,PreRetCorr]=stk_return_forecast(model_data,PreBeta,factor_list2,3)

