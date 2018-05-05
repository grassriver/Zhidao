# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:09:45 2018

@author: Hillary
"""
import pandas as pd
import sqlite3 as sql
import numpy as np
import datetime as dt
import general_tools as tool
#import data_formating as dft

import sys
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On')
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model')



#%%
conn=sql.connect('C:/Users/Hillary/Documents/PI/data/data.db')

#start = '2015-01-01'
#end = '2016-12-31'

start = '2014-01-01'
end = '2016-12-31'

holdout_start = '2017-01-01'
holdout_end = '2017-12-30'

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
        
        # pull data from fundamental table
        fundamental = pd.read_sql("select distinct * from fundamental where date>='"+min_quarter+"' and date <='"+max_quarter+"'",conn)
        fundamental['date']=pd.to_datetime(fundamental['date'])
        fundamental['quarter']=fundamental['date'].dt.quarter
        fundamental['year']=fundamental['date'].dt.year
                
        # merge stock price with fundamental information
        fundamental_merge=pd.merge(quarter_end,fundamental,left_on=['code','year','quarter'],right_on=['code','year','quarter'],how='inner')
        fundamental_merge['marketcap'] = fundamental_merge['close']*fundamental_merge['outstanding']
        return fundamental_merge, stock_prices
    
[fundamental_merge,stock_prices]=get_fundamental(conn,start,end)  

#%% factor generator


def fundamental_factor_generator(df):
    
    df['size'] = np.log(df['marketcap'])
    df['BM'] = df['bvps']/df['close']
    df['leverage'] = ((df['epcf']*df['outstanding'])/df['cf_liabilities'])/df['marketcap']
                
    df=df.sort_values(by=['code','year','quarter']) 
    df['roe_gr']=df.groupby('code',as_index=False)['roe'].apply(lambda x: (x/x.shift(1))-1)
    df['eps_gr']=df.groupby('code',as_index=False)['eps'].apply(lambda x: (x/x.shift(1))-1)
    
    return df

fundamental_merge=fundamental_factor_generator(fundamental_merge)

def technical_factor_generator(df1,df2,start,end):
    
    df =pd.read_sql("select * from technical where date >='"+start+"' and date <= '"+end+"' order by code,date asc",conn)
    df['date']=pd.to_datetime(df['date'])
    df=df.sort_values(by=['code','date']) 
    df['year']=df['date'].dt.year
    df['quarter']=df['date'].dt.quarter
    df['volatility']=(df['20HIGH']-df['20LOW'])/(df['20HIGH']+df['20LOW'])
    df['avg_volume'] = df1.groupby('code',as_index=False)['volume'].rolling(window=20,min_periods=1,center=False).mean()
    temp = df2['code','year','quarter','marketcap']
    df = pd.merge(df,temp, on=['code','year','quarter'])
    df['liquidity']=df['avg_volume']/df['marketcap']
    df.drop('marketcap')
    
    return df


technical = technical_factor_generator(stock_prices,fundamental_merge,start,end)    
    
#################### Fundamental Factors ###############################      
#%% summary statistics of included factors

factor_list = pd.read_csv('C:/Users/Hillary/Documents/PI/Doc/Stock Screener/mapping_factors.csv')
fundamental=factor_list[factor_list['tables']=='fundamental']
add = pd.DataFrame([['marketcap','fundamental','quarterly']],columns=['variable','tables','frequency'])
fundamental=pd.concat([add,fundamental])
fundamental=fundamental[fundamental.variable !='distrib']

keeplist = list(fundamental['variable'])

key = ['code','year','quarter']

for i in key:
    keeplist.append(i)

fundamental_merge[list(fundamental['variable'])]=fundamental_merge[list(fundamental['variable'])].apply(pd.to_numeric,errors='coerce')

sum_stat=fundamental_merge[list(fundamental['variable'])].describe()       

period = '_'+start+'_'+end
sum_stat.to_csv("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_summary_statistics"+period+".csv")

#%% Fill missing values

#before_fill=pd.DataFrame()
#before_fill.to_csv("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_missing_values"+period+".csv")

# mannually copy the .info() results to csv file
#fundamental_merge[list(fundamental['variable'])].info()


del fundamental_merge['distrib']

def missing_sum(df,varlist):
    
    for ff in varlist:
        df[ff][df[ff]==np.inf]=np.nan
    
    miss = df.copy()
    miss[varlist]=miss[varlist].isnull()
    return miss
    
miss_sum = missing_sum(fundamental_merge,list(fundamental['variable']))

miss_report = miss_sum.groupby(['year','quarter'])[list(fundamental['variable'])].apply(lambda x: x.sum()/x.count())    

miss_report.to_csv("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_miss_summary"+period+".csv")

def list_obs(df,year,quarter,factor,key='code'):
    
    show =df[((df['year']==year) & (df['quarter']==quarter))][['year','quarter',key,factor]]
    show

tt=list_obs(fundamental_merge,2000,1,'roe')   

miss_report_agg = miss_report[list(fundamental['variable'])].apply(lambda x: x[x>0.1].count())

drop_list = pd.DataFrame(miss_report_agg[miss_report_agg>15]).reset_index().loc[:,'index']


"""
# fill missing values

def fill_missing(df, varlist, by1, by2, key):
    
    '''
    fill missing value using the following waterfall logic: 
      first by the median of 'by1' group
      then if still missing&by2 is specified, fill by the median of 'by2' group

    '''
    

   # missing_matrix = df.copy()
    #missing_matrix[varlist] = df[varlist].isnull()    

    df[varlist]=df.groupby(by1)[varlist].transform(lambda x: x.fillna(x.median()))
    

    if len(by2)==0:
          
        return df, missing_matrix
       
    elif len(by2)>0:
        
        df[varlist]=df.groupby(by2)[varlist].transform(lambda x: x.fillna(x.median()))
    
        return df, missing_matrix

[fundamental_merge,fundamental_miss]=fill_missing(fundamental_merge,list(fundamental['variable']),'code',['year','quarter'],['code','year','quarter'])

"""
fundamental_miss.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_missing"+period+".pkl")

#fundamental_miss1=pd.DataFrame(np.zeros(fundamental_miss.shape))

                 
#%%



#%% Normalization


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

#### for the current run, 1% and 99% winsorization of universe is used ### 
def data_preprocess(df,varlist,outlier_by=[],normalize_by=['year','quarter']):
    
    df = outlier_method(df,varlist,by=[])
    
    for ff in varlist:
    
       df[ff] = df.groupby(normalize_by)[ff].apply(normalize_series)
       
    return df   
    
    
fundamental_merge = data_preprocess(fundamental_merge,list(fundamental['variable']))
fundamental_merge=fundamental_merge[keeplist]

fundamental_merge.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_fill"+period+".pkl")

sum_stat_fill=fundamental_merge[list(fundamental['variable'])].describe()       
sum_stat_fill.to_csv("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_summary_stat_normalize"+period+".csv")

# hold-out period

[fundamental_holdout,stock_prices_holdout]=get_fundamental(conn,holdout_start,holdout_end)  

fundamental_holdout=fundamental_holdout[keeplist]

holdout_period =  '_'+holdout_start+'_'+holdout_end
fundamental_holdout.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_holdout_raw"+holdout_period+".pkl")

################### Stock Returns #########################

stk_keep_list = ['code','date','close']

stock_prices_main = stock_prices[stk_keep_list]
stock_prices_holdout_main =stock_prices_holdout[stk_keep_list]

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

stk_list = ['stock_prices_main','stock_prices_holdout_main']

model_ret_day = log_ret(stock_prices_main,freq='day',key='code',date_var='date')
holdout_ret_day = log_ret(stock_prices_holdout_main,freq='day',key='code',date_var='date')
   

  
'''

sl_dict = {(s1:'stock_prices_main'),}

for sl in ['stock_prices_main','stock_prices_holdout_main']:
    for fl in ['day','week','month']:
        temp= sl+'_'+fl
        temp = log_ret(stock_prices_holdout_main,freq=fl,key='code',date_var='date')
'''        

################## merge stock return with fundamental factors #################

def merge_fundamental()

model_data = pd.merge(model_ret_day,fundamental_merge,how='left',on=['code','year','quarter'])  

holdout_data = pd.merge(holdout_ret_day,fundamental_holdout,how='left',on=['code','year','quarter'])      

model_data['date']=model_data['date'].dt.strftime('%Y-%m-%d')
holdout_data['date']=holdout_data['date'].dt.strftime('%Y-%m-%d')

model_data.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/model_data_daily"+period+".pkl")

holdout_data.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/holdout_data_daily"+holdout_period+".pkl")

    
        

