# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 21:38:41 2018

@author: Hillary
"""

import pandas as pd
import sqlite3 as sql
import numpy as np
import datetime as dt
#import data_formating as dft

import sys
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On')
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model')

from portfolio_class import Portfolio
import general_tools as tool


#%%
conn=sql.connect('C:/Users/Hillary/Documents/PI/data/data.db')

#start = '2015-01-01'
#end = '2016-12-31'

start = '2001-01-01'
end = '2017-12-31'

#holdout_start = '2017-01-01'
#holdout_end = '2017-12-30'

#%% generate fundamenal factor data
def get_fundamental(conn, start, end):
     
        '''
        pull stock price data and fundamental factor data
        merge fundamental table with stock price of the last day of the corresponding quarter
        to calculate market capitalization and keep all fundamental factors
     
        '''
        stock_prices = pd.read_sql("select a.code,a.date,a.close,a.volume,b.industry,b.area,b.outstanding,b.timetomarket from stocks_price a left join stock_basics b on a.code = b.code where date >='"+start+"' and date <= '"+end+"' order by a.date,a.code asc",conn)
        stock_prices['qtr_range'] = stock_prices.date.apply(tool.get_lagged_time,freq = 'quarterly',lag = 0)
        min_quarter = stock_prices.qtr_range.min()
        max_quarter= stock_prices.qtr_range.max()
        
        stock_prices['date']=pd.to_datetime(stock_prices['date'])
        stock_prices['quarter']=stock_prices['date'].dt.quarter
        stock_prices['year']=stock_prices['date'].dt.year
        stock_prices = stock_prices.sort_values(by=['code','date']) 
        
        # keep last day of each quarter 
        quarter_end=stock_prices.copy()
        quarter_end = quarter_end.groupby(['code','year','quarter'],as_index=False).last()
        #quarter_end=quarter_end.reset_index()           
        
        #dt.quarter get quarter end
        # pull data from fundamental table
        fundamental = pd.read_sql("select distinct * from fundamental where date>='"+min_quarter+"' and date <='"+max_quarter+"'",conn)
        fundamental['date']=pd.to_datetime(fundamental['date'])+pd.offsets.QuarterEnd()
        fundamental['quarter']=fundamental['date'].dt.quarter
        fundamental['year']=fundamental['date'].dt.year
                
        # merge stock price with fundamental information
        fundamental_merge=pd.merge(quarter_end,fundamental,left_on=['code','year','quarter'],right_on=['code','year','quarter'],how='inner')
        fundamental_merge['marketcap'] = fundamental_merge['close']*fundamental_merge['outstanding']
        return fundamental_merge, stock_prices
    
[fundamental,stock_prices]=get_fundamental(conn,start,end)  

fundamental_merge=fundamental.copy()

fundamental_merge['adratio']=pd.to_numeric(fundamental_merge['adratio'],'coerce')
                

#%% factor generator

fundamental_list = ['size','BM','leverage',
           'PE','marketcap',
           'mbrg','nav','net_profit_ratio','bvps'] 


def fundamental_factor_generator(df):
    
    df['size'] = np.log(df['marketcap'])
    df['BM'] = df['bvps']/df['close']
                
    df=df.sort_values(by=['code','year','quarter']) 
    #df['roe_gr']=df.groupby('code')['roe'].apply(lambda x: (x/x.shift(1))-1)
    #df['eps_gr']=df.groupby('code')['eps'].apply(lambda x: (x/x.shift(1))-1)
    
    df=df.rename(columns={'adratio':'leverage'})

    return df

fundamental_merge=fundamental_factor_generator(fundamental_merge)

#%%#################### Get industry classification data #############
industry = pd.read_excel('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/industry_classification2.xlsx')
industry.columns = ['code','name','drop1','drop2','drop3','wind']
industry = industry.drop(['drop1','drop2','drop3'],axis=1)

industry['code']=industry['code'].replace({'.SZ':''},regex=True)
industry['code']=industry['code'].replace({'.SH':''},regex=True)


industry_code = industry['wind'].to_frame().drop_duplicates()

industry_code['industry_code']=np.arange(len(industry_code))

industry = pd.merge(industry,industry_code)

industry_mapping = industry.drop(['name'],axis=1)

industry_mapping = industry_mapping[industry_mapping.industry_code !=11]


#%%################ add industry mapping to fundamental table ##########
fundamental_merge = pd.merge(fundamental_merge,industry_mapping,right_on='code',left_on='code')

#fill missing values
def fill_miss(df, varlist,by1,by2,by3):
    
    for ff in varlist:
        df[ff][df[ff]==np.inf]=np.nan
          
    df[varlist]=df.groupby(by1,as_index=False)[varlist].transform(lambda x: x.fillna(x.median()))
    
    if len(by2)>0:
        
            df[varlist]=df.groupby(by2,as_index=False)[varlist].transform(lambda x: x.fillna(x.median()))

            if len(by3)>0:
                 
                 df[varlist]=df.groupby(by3,as_index=False)[varlist].transform(lambda x: x.fillna(x.median()))

    
    return df


fundamental_merge=fill_miss(fundamental_merge,fundamental_list,'code',['year','quarter','industry_code'],['industry_code'])

#%%################## Outlier cleaning ############################


def winsorize_series(s,pctl):
    
        q = s.quantile([pctl, 1-pctl])
        s[s < q.iloc[0]] = q.iloc[0]
        s[s > q.iloc[1]] = q.iloc[1]
        
        return s
    
def standardize_series(s,num):
    
        mean, std = np.nanmean(s), np.nanstd(s)
        r_outliers = s > mean+num*std
        l_outliers = s < mean- num*std            
        s[r_outliers] = mean+num*std
        s[l_outliers] = mean-num*std    
         
        return s
    
        
def outlier_method(df,varlist,by=[]):
    
    '''
    handling outliers based on chosen method:
    method: 'win' for winsorization
            'std' for # std distance from mean
    'by' is optional        
    
    '''
    
    method = input('use win or std: ')
    
    if method =='std':
        num = float(input('enter the number of std from mean: '))
        if len(by)>0:
            for ff in varlist:
                df[ff]=df.groupby(by)[ff].apply(standardize_series,num=num)
        elif len(by)==0:
            for ff in varlist:
                df[ff]=standardize_series(df[ff],num=num)
        
       
    elif method =='win':
        pctl = float(input("enter the percentile (in decimal) for winsorization: "))
        if len(by)>0:
            for ff in varlist:
                df[ff]=df.groupby(by)[ff].apply(winsorize_series,pctl=pctl)
        elif len(by)==0:
            for ff in varlist:
                df[ff]=winsorize_series(df[ff],pctl=pctl)
            
    return df      

def normalize_series(s):
    

    mean, std = np.nanmean(s), np.nanstd(s)
    
    s= (s-mean)/std
       
    return s

#%%### for the current run, 3 std from the mean is used ### 
def data_preprocess(df,varlist,outlier_by=[],normalize_by=['year','quarter']):
    
    for ff in varlist:
        df[ff][df[ff]==np.inf]=np.nan
    
    df = outlier_method(df,varlist,by=[])
    
    for ff in varlist:
    
       df[ff] = df.groupby(normalize_by)[ff].apply(normalize_series)
       
    return df   

fundamental_merge = data_preprocess(fundamental_merge,fundamental_list)


#%%################# technical indicator ######################

technical_list = ['code','date','year','quarter','Beta','volatility','liquidity','MACD','GAP']


def technical_factor_generator(df1,df2,start,end):
    
    df =pd.read_sql("select * from technical where date >='"+start+"' and date <= '"+end+"' order by code,date asc",conn)
    df['date']=pd.to_datetime(df['date'])
    df=df.sort_values(by=['code','date']) 
    df['year']=df['date'].dt.year
    df['quarter']=df['date'].dt.quarter
    df['volatility']=(df['20HIGH']-df['20LOW'])/(df['20HIGH']+df['20LOW'])
    df1_temp=df1[['code','date','volume']]
    df2_temp=df2[['code','year','quarter','marketcap']]
    df = pd.merge(df,df1_temp,on=['code','date'])
    df = pd.merge(df,df2_temp,on=['code','year','quarter'])
    df=df.sort_values(by=['code','date'])
    df['avg_volume'] = df.groupby('code')['volume'].apply(lambda x: x.rolling(window=20,min_periods=1).mean())
    df['liquidity']=df['avg_volume']/df['marketcap']
    #df= df.drop('marketcap',axis=1)
    df=df[technical_list]
    
    return df


technical = technical_factor_generator(stock_prices,fundamental_merge,start,end) 

#%%###################### get market return  #################################

stocks_list = pd.DataFrame({'code': ['000002', '000001'], 'shares': [100, 100]})
p = Portfolio(conn, stocks_list, start=start, end=end)

#p.add_benchmark()

mkt_ret = p.benchmark_returns()
mkt_ret = mkt_ret.reset_index()
mkt_ret = mkt_ret.rename(columns={'index': 'mkt_ret'})


#%%################ stock returns #################
def freq_converter(df,date_var,key,freq):
    
    '''
    valid freq inputs: week, month, quarter
    
    '''
    df[date_var]=pd.to_datetime(df[date_var])
    df['year']=df[date_var].dt.year
    df['quarter']=df[date_var].dt.quarter
                  
    if freq=='week':
        df['week']=df[date_var].dt.week
        df = df.sort_values(by=[key,date_var])  
        df = df.groupby([key,'year','week'],as_index=False).last()
    elif freq=='month':
        df['month']=df[date_var].dt.month
        df = df.sort_values(by=[key,date_var])  
        df = df.groupby([key,'year','month'],as_index=False).last()
    elif freq=='quarter':    
        df = df.sort_values(by=[key,date_var])  
        df = df.groupby([key,'year','quarter'],as_index=False).last()

    return df


def log_ret(df,freq='day',key='code',date_var='date'):
    
    '''
    calculate individual stock log return based on chosen frequency
    valid freq inputs are: day, week, month
    '''
    df = freq_converter(df,date_var,key,freq)
    df = df.sort_values(by=date_var)
    df['log_ret']=df.groupby('code')['close'].apply(lambda x: np.log(x)-np.log(x).shift(1))
    
    return df        

#daily return
stock_prices_main = stock_prices[['code','date','close']]
model_ret_day = log_ret(stock_prices_main,freq='day',key='code',date_var='date')

#check_stock =  model_ret_day[model_ret_day['code']=='600021'][['code','date','close','log_ret']]

#check_stock.to_csv('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/test_return_600021.csv')

##################### Merge Data using lag1 for now ##################
#%%
#technical['m_date']=technical['date']+dt.timedelta(days=1)
technical=technical.sort_values(by=['code','date'])
technical['m_date']=technical.groupby('code',as_index=False)['date'].shift(-1)
technical =technical.drop(['date','year','quarter'],axis=1)

model_data = pd.merge(model_ret_day,technical,copy=False, 
                      left_on=['code','date'],right_on=['code','m_date'],how='inner')

model_data=model_data.drop(['close','m_date'],axis=1)   

# merge fundamental factors
#fundamental_merge['m_date'] = fundamental_merge['date_y']+dt.timedelta(days=91)
fundamental_merge=fundamental_merge.sort_values(by=['code','date_y'])
fundamental_merge['m_date']=fundamental_merge.groupby('code',as_index=False)['date_y'].shift(-1)

fundamental_merge['m_quarter']=fundamental_merge['m_date'].dt.quarter
fundamental_merge['m_year']=fundamental_merge['m_date'].dt.year
fundamental_merge=fundamental_merge.drop(['year','quarter','date_x','date_y'],axis=1)
                 
model_data = pd.merge(model_data,fundamental_merge,left_on=['code','year','quarter'],
                      right_on = ['code','m_year','m_quarter'],how='inner')                 

# merge check
#merge_check1 = model_data.groupby('date',as_index=False)['code'].count()

model_data=model_data.drop(['level_0','m_date','m_year','m_quarter'],axis=1)

# add market return
#mkt_ret['m_date']=mkt_ret['date']+dt.timedelta(days=1)
mkt_ret=mkt_ret.sort_values(by='date')
mkt_ret['m_date']=mkt_ret.date.shift(-1)

model_data = pd.merge(model_data,mkt_ret,left_on=['date'],right_on=['m_date'])

model_data = model_data.drop(['m_date','date_y'],axis=1)
model_data = model_data.rename(columns={'date_x': 'date'})

# merge check
#merge_check2 = model_data.groupby('date',as_index=False)['code'].count()



varlist = ['MACD','GAP','liquidity','Beta','size','BM','leverage',
           'PE','volatility','marketcap',
           'mbrg','nav','net_profit_ratio','bvps','mkt_ret']

#var_pct=['adratio','roe','mbrg','nav','net_profit_ratio']

# generate dummies for industries based on ind_code
#model_data=model_data.rename(columns={'ind_code':'industry_code'})

model_data['industry_code']=model_data['industry_code'].astype(str)

td=pd.get_dummies(model_data['industry_code'],dummy_na=False,prefix='ind',drop_first =True)

model_data=model_data.join(td)

model_data['country']=1

period = '_'+start+'_'+end

#model_data.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/modeling_data"+period+".pkl")

model_data.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/modeling_data_wind"+period+".pkl")

ind_list = model_data.columns[pd.Series(model_data.columns).str.startswith('ind_')]




