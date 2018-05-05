# -*- coding: utf-8 -*-
# Factor Model
"""
Created on Fri Apr 13 11:58:30 2018

@author: Hillary


"""

import sqlite3 as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas.tseries.offsets import BDay

import statsmodels.tsa as ts
import statsmodels.api as sm

#import sklearn.metrics as metrices 
#import sklearn.linear_model as lm
import sys
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On')
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model')
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Time_Series')

import general_tools as tool
import test_ts_dist 
from arma_garch import ARMA_GARCH

conn1 = 'C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/modeling_data_wind_2001-01-01_2017-12-31.pkl'


model_data = pd.read_pickle(conn1)

insample = model_data[(model_data['date']>='2001-01-01')&(model_data['date']<='2017-12-31')]

#outsample = model_data[(model_data['date']>='2017-01-01')&(model_data['date']<='2017-12-31')]


ind_list = (model_data.columns[pd.Series(model_data.columns).str.startswith('ind_')]).tolist()

varlist = ['MACD','GAP','liquidity','size','BM','leverage',
           'PE','roe_gr','eps_gr','volatility','marketcap',
           'cf_liabilities','cf_sales','roe',
           'mbrg','nav','net_profit_ratio','bvps']

#factor_list1 = ['liquidity','size','leverage',
                'volatility','MACD',]+list(ind_list)

factor_list2 = ['liquidity','size','leverage','Beta','country',
                'volatility','MACD',]+list(ind_list)


#keep = ['code','year','date','log_ret']+factor_list1
       
keep2 = ['code','year','date','log_ret']+factor_list2
     

data1 = insample[keep2]

#data2 = outsample[keep]


#%%#################### Cross-section of Beta : Two-step Approach ################################# 

'''
tt=data1[data1['date']=='2011-01-04']
tt.dropna(inplace=True)

tt=tt.set_index(['code','date'],drop=False)


X=tt.loc[:,list(ind_list)]
X=sm.add_constant(X)

results = sm.OLS(tt.log_ret,X).fit()

wls_model = sm.WLS(tt.log_ret,X,weights =tt.size).fit()

t1=wls_model.params.to_frame()

df = data1[data1['year']==2016]


df.dropna(inplace=True)
'''

  
# Step 1. OLS Regression 


def OLS_reg(y,x):
    
    x=sm.add_constant(x)
    
    model = sm.OLS(y,x).fit()
    
    ols_params =pd.DataFrame([model.params])
    
    ols_adjr2 = pd.DataFrame([model.rsquared_adj])
    ols_adjr2.columns=['adjr2']
    
    ols_r2 = pd.DataFrame([model.rsquared])
    ols_r2.columns=['r2']

    ols_resid = (model.resid).to_frame()
    ols_resid = ols_resid.reset_index()
    #ols_resid.columns=['code','date','ols_res']
    #ols_yhat = results.predict()
    
    return ols_params, ols_adjr2, ols_r2, ols_resid

#[beta,adjr2,r2, ols_resid]=OLS_reg(tt.log_ret,tt[factor_list2])


def beta_s1(df,index,y,x):
    
    beta, adjr2, r2, resid =  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df.dropna(inplace=True)
    
    df = df.set_index(index,drop=False)
    
    df = df.sort_values(by='date')
    
    for t in df.date.unique():
        
        print(t)
        
        [params,ar2,r,res]=OLS_reg(df.loc[df.date==t,:][y],df.loc[df.date==t,:][x])
        
        params['date']=t
        ar2['date']=t
        r['date']=t   
           
        beta  =beta.append(params)
        adjr2 =adjr2.append(ar2)
        r2 = r2.append(r)
        resid = resid.append(res)
        
    return beta, adjr2, r2, resid   

'''
[ols_beta,ols_adjr2,ols_r2,ols_resid]=beta_s1(data1,['date','code'],'log_ret',factor_list1)

ols_beta.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/OLS_Beta.csv')

ols_adjr2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/OLS_Adjr2.csv')

ols_r2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/OLS_r2.csv')

ols_resid = ols_resid.rename(columns={0: 'ols_resid'})

ols_resid.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/OLS_resid.csv')

[ols_beta,ols_adjr2,ols_r2,ols_resid]=beta_s1(data1,['date','code'],'log_ret',factor_list1)

ols_beta.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_OLS_Beta.csv')

ols_adjr2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_OLS_Adjr2.csv')

ols_r2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_OLS_r2.csv')

ols_resid = ols_resid.rename(columns={0: 'ols_resid'})

ols_resid.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_OLS_resid.csv')

'''
[ols_beta,ols_adjr2,ols_r2,ols_resid]=beta_s1(data1,['date','code'],'log_ret',factor_list2)

ols_beta.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_OLS_Beta.csv')

ols_adjr2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_OLS_Adjr2.csv')

ols_r2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_OLS_r2.csv')

ols_resid = ols_resid.rename(columns={0: 'ols_resid'})

ols_resid.to_pickle('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_OLS_resid.pkl')




'''

data_out =data1[data1['date']>='2015-01-01']

data_out.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_model_data_2015_2016.csv')
'''
       
# Step 2. WLS Regression

ols_resid['ols_resid2']=ols_resid['ols_resid']*ols_resid['ols_resid']

weight = ols_resid.groupby('code')['ols_resid2'].apply(lambda x: x.sum()/(x.count()-1)).to_frame()

weight=weight.reset_index()

wls_df = pd.merge(data1,weight,on=['code'])


def WLS_reg(y, x, weight):

    x=sm.add_constant(x)
    wls_model = sm.WLS(y,x,weights=weight).fit()

    wls_params = pd.DataFrame([wls_model.params])
    
    wls_adjr2 = pd.DataFrame([wls_model.rsquared_adj])
    wls_adjr2.columns=['adjr2']
    
    wls_r2 = pd.DataFrame([wls_model.rsquared])
    wls_r2.columns=['r2']
    
    wls_resid = (wls_model.resid).to_frame()
    wls_resid = wls_resid.reset_index()
    
    wls_yhat = (wls_model.fittedvalues).to_frame()
    wls_yhat = wls_yhat.reset_index()
    

    return wls_params, wls_adjr2, wls_r2, wls_resid, wls_yhat

#[beta2,adjr22,r2,wls_resid,wls_yhat]=WLS_reg(tt.log_ret,tt[factor_list1],tt.ols_resid2)

def beta_s2(df,index,y,x, weight):
    
        
    beta, adjr2, resid, yhat, r2 =  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df.dropna(inplace=True)
    
    df = df.set_index(index,drop=False)
    
    df = df.sort_values(by='date')
    
    for t in df.date.unique():

        print(t)
        
        [params,ar2,r,res,yt]=WLS_reg(df.loc[df.date==t,:][y],df.loc[df.date==t,:][x],df.loc[df.date==t,:][weight])
        
        params['date']=t
        ar2['date']   =t
        r['date']     =t
           
        beta  =beta.append(params)
        adjr2 =adjr2.append(ar2)
        r2 =r2.append(r)
        resid = resid.append(res)
        yhat = yhat.append(yt)
        
    return beta, adjr2, r2,resid, yhat   

[wls_beta,wls_adjr2,wls_r2, wls_resid,wls_yhat]=beta_s2(wls_df,['date','code'],'log_ret',factor_list2,'ols_resid2')

'''
wls_beta.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/WLS_Beta.csv')

wls_adjr2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/WLS_adjr2.csv')

wls_r2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/WLS_r2.csv')
'''

'''

wls_beta.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_WLS_Beta.csv')

wls_adjr2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_WLS_adjr2.csv')

wls_r2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/wind_WLS_r2.csv')

'''
# calculate residual variance

wls_resid = wls_resid.rename(columns={0: 'wls_resid'})

wls_resid['wls_resid2']=wls_resid['wls_resid']*wls_resid['wls_resid']


wls_beta.to_pickle('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_WLS_Beta.pkl')

wls_adjr2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_WLS_adjr2.csv')

wls_r2.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_WLS_r2.csv')

wls_resid.to_pickle('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_WLS_resid_var.pkl')

wls_yhat = wls_yhat.rename(columns={0: 'wls_yhat'})
wls_yhat.to_pickle('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/Country&Beta_WLS_yhat.pkl')


#%%##################### calculate time-series of Rsquared ##############################

tsr = pd.merge(wls_resid,wls_yhat,on=['code','date']).drop_duplicates()

def ts_rsq(df,yhat,resid2):
    
    return 1-(df[resid2].sum())/(((df[yhat]-df[yhat].mean())**2).sum())

def ts_adjrsq(df,yhat,resid2,r2):
    
    return 1-((1-df[r2])*(yhat.count()-1)/((yhat.count() - len(factor_list2) -1)))
    
    
    
ts_r2= tsr.groupby('code').apply(ts_rsq,'wls_yhat','wls_resid2').reset_index()
#%%########################## Modeling Time-Series of Betas ##############################

# test stationarity and autocorrelation of in-sample betas
def time_series_stationarity(df,factor_list):
    
    df = df.sort_values(by='date')
    
    for ff in factor_list:
       
       print ('Dickey-Fuller test for factor %s'%ff)
       test_df = test_ts_dist.test_stationarity(df[ff])
       
       print('Ljung-Box test on autocorrelation for factor %s'%ff)
       test_ljq = test_ts_dist.ljungbox_test(df[ff]) 
       
time_series_stationarity(wls_beta, factor_list2)   


#HWNote: visual check shows betas for industry dummies are  
#        populated wiht zeros before 2001 July 

wls_beta_use = wls_beta[wls_beta['date']>='2001-07-01']

#Set forecast period

f_period=5

def beta_ARMA_GARCH(df,model,factor_list,f_period):
    
    beta_forecast = pd.DataFrame()
    beta_conditional_vol = pd.DataFrame()
    
    for ff in factor_list:
        
        print('modeling for factor beta %s: '%ff)
        
        model.estimation(df[ff].values)
        model.prediction(f_period)
        
        beta_forecast[ff]=model.prediction_x
        beta_conditional_vol[ff] = model.prediction_conditional_vol
        beta_forecast['f_period']=np.arange(1,f_period+1)                    
                     
    return beta_forecast,   beta_conditional_vol      

[PreBeta,PreBetaVol] = beta_ARMA_GARCH(wls_beta_use, ARMA_GARCH(order1=[1,0,1],order2=[1,1]),factor_list1,f_period)     


#Historical Beta Correlation Matrix

wls_beta_demean = wls_beta_use.copy()

wls_beta_demean[factor_list1] = wls_beta_demean[factor_list1].apply(lambda x: x-x.mean())

HistBetaCov =pd.DataFrame((np.matrix(wls_beta_demean[factor_list1]).T*
                           np.matrix(wls_beta_demean[factor_list1]))/(wls_beta_demean.shape[0]-1),
                           columns =factor_list1 )
    
HistBetaVar = pd.DataFrame({'HistBetaVar':np.diagonal(np.matrix(HistBetaCov))})

HistBetaVar['HistBetaStd']=HistBetaVar['HistBetaVar'].apply(np.sqrt)

HistBetaVar['beta'] = pd.Series(factor_list1)

HistBetaStdMul = np.matrix(HistBetaVar.HistBetaStd).T*np.matrix(HistBetaVar.HistBetaStd)

HistBetaCorr = pd.DataFrame(np.divide(np.matrix(HistBetaCov),HistBetaStdMul),
                            columns=factor_list1)

HistBetaCorr['beta'] = pd.Series(factor_list1)


#%%############# predict covariance matrix of the returns under BARRA approach ##########

#1 Stock Return Volatility Forecasts


def forecast_feed(df, factor_list):
    
    feed =df[df.date == df.date.min()]
        
    feed[factor_list]=feed[factor_list].transform(lambda x:x.fillna(x.median()) )
    
    feed = feed.sort_values(by='code').drop_duplicates(subset='code')
    
    code_list = feed.code.tolist()
    
    #anchor_date = df.date.min()
    
    return feed, code_list
  
[forecast_feed, code_list] = forecast_feed(data2,factor_list1)

  
def forecast_beta_cov(PreBetaVol, HistBetaCorr, f_day):
    
    if f_day>len(PreBeta):
        raise ValueError('!!!chosen forecast period is longer than the series of beta forecasts') 

    row = f_day -1 
    
    PreBetaStdMul = np.matrix(PreBetaVol.iloc[row,:]).T*np.matrix(PreBetaVol.iloc[row,:])

    PreBetaCov = pd.DataFrame(np.multiply(np.matrix(HistBetaCorr[factor_list1]),PreBetaStdMul),
                          columns = factor_list1)
    
    return PreBetaCov


def forecast_ret_cov1(feed, BetaCov,code_list):
    
    PreRetCov = pd.DataFrame(np.matrix(feed[factor_list1]) 
                         *np.matrix(BetaCov[factor_list1]) 
                         *np.matrix(feed[factor_list1]).T ,columns =code_list )

    PreRetCov['code'] = pd.DataFrame(code_list)
    
    return PreRetCov


def forecast_ret_cov2(feed,factor_list,PreBetaVol,HistBetaCorr,f_day):
    
    '''
    f_day: the ith day of forecast
    feed: factor values used as forecast input
    PreBetaVol: beta variance forecasts from ARMA-GARCH module
    HistBetaCorr: historical beta correlation matrix
    
    '''
        
    PreBetaCov = forecast_beta_cov(PreBetaVol, HistBetaCorr,f_day)

    PreRetCov = forecast_ret_cov1(forecast_feed, PreBetaCov, code_list)

    #start_date = anchor_date - BDay(252)
    start_date = wls_resid.date.max() - dt.timedelta(days=365)

    resid_var_in =wls_resid.loc[(wls_resid['code'].isin(code_list))&(wls_resid['date']>=start_date)]
    
    resid_var=pd.DataFrame({'resid_var':resid_var_in.groupby('code')['wls_resid2'].apply(lambda x: x.sum()/(x.count()-1))})

    resid_var = resid_var.reset_index()    
    
    PreRetCov = pd.merge(PreRetCov, resid_var, on='code',how='left')
    
    PreRetCov['resid_var']=PreRetCov['resid_var'].transform(lambda x:x.fillna(x.median()) )
    
    for i in range(len(PreRetCov)):
        PreRetCov.iloc[i,i] = PreRetCov.iloc[i,i]+PreRetCov.loc[i,'resid_var']
    
    PreRetCov = PreRetCov.drop('resid_var',axis=1) 
   
    return PreRetCov

PreRetCov = forecast_ret_cov2(forecast_feed,factor_list1,PreBetaVol,HistBetaCorr,1)
 
   
#np.diagonal(np.matrix(PreRetCov[PreRetCov.columns[PreRetCov.columns !='code']])*252 )   

PreRetCov.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/FM_PreRetCov_2017_01_03.pkl")
   

'''
#2.2 Use historical Beta Cov from last quarter

HistBetaCov = wls_beta[wls_beta['date']>='2016-10-01'][factor_list1].cov()

HistBetaCov['beta'] = pd.Series(factor_list1)

PreRetCov_hist = pd.DataFrame(np.matrix(forecast_feed[factor_list1]) 
                         *np.matrix(HistBetaCov[factor_list1]) 
                         *np.matrix(forecast_feed[factor_list1]).T ,columns =code_list )

PreRetCov_hist['code'] = pd.Series(code_list)
'''


# Interpolate missing values
# PreRetCov = PreRetCov.interpolate()

#2 Stock Return Forecasts, not used, for referene only

Intercept = wls_beta_use[wls_beta_use['date']==wls_beta.date.max()].const

def stk_return_forecast(load,PreBeta,factor_list,f_period):
    
    if f_period>len(PreBeta):
        raise ValueError('!!!chosen forecast period is longer than the series of beta forecasts') 
        
    use = factor_list + ['code']
    
    load = load[use]
    load = load.set_index('code',drop=False)
    
    PreReturn = pd.DataFrame([])
    
    for t in range(f_period):
              
        ret = pd.DataFrame({'PreRet':load[factor_list].dot(PreBeta.iloc[t][factor_list])})+Intercept[0]
                
        ret=ret.reset_index()
        t+=1
        ret['f_period']=t
        
        PreReturn = PreReturn.append(ret)
           
        PreReturn['PreRet'] = PreReturn.groupby(by='f_period',as_index=False)['PreRet'].transform(lambda x: x.fillna(x.median()))
   
    PreReturn = PreReturn.drop_duplicates()
    
    #pivot_ret =PreReturn.pivot_table(index='f_period',columns='code',values='PreRet') 
        
    return PreReturn



PreRet=stk_return_forecast(forecast_feed,PreBeta,factor_list1,f_period)



#%%###########output Return and Covariance Forecasts######################

PreRet.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/FM_PreReturn.pkl")
PreRetCov.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/FM_PreRetCov.pkl")


#%%### time-series regression r-squared

ret_data = data1[['code','date','log_ret']]
test_data = pd.merge(ret_data,wls_beta,on='date',how='inner')      

  
def OLS_adjr2(df,xcols,ycol):
        
    return sm.OLS(df[ycol],df[xcols]).fit().rsquared_adj

ts_adjr2 =pd.DataFrame({'adjr2': test_data.groupby('code').apply(OLS_adjr2,factor_list1,'log_ret')})


test_data[test_data['code']=='000333'].count()
#%%








