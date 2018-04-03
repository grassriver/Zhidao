# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:09:45 2018

@author: Hillary
"""
import pandas as pd
import sqlite3 as sql
import numpy as np
import datetime as dt

import sys
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On')
sys.path.append('C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model')

#import general_tools as tool
import data_formating as dft

#%%
conn=sql.connect('C:/Users/Hillary/Documents/PI/data/data.db')

start = '2015-01-01'
end = '2016-12-31'

#start = '2000-01-01'
#end = '2018-12-31'

#%% generate fundamenal factor data
def get_fundamental(conn, start, end):
     
        '''
        pull stock price data and fundamental factor data
        merge fundamental table with stock price of the last day of the corresponding quarter
        to calculate market capitalization and keep all fundamental factors
     
        '''
        stock_prices = pd.read_sql("select a.*,b.industry,b.area,b.outstanding,b.timetomarket from stocks_price a left join stock_basics b on a.code = b.code where date >='"+start+"' and date <= '"+end+"' order by a.date,a.code asc",conn)
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
        return fundamental_merge
    
fundamental_merge=get_fundamental(conn,start,end)        
#%% summary statistics of included factors

factor_list = pd.read_csv('C:/Users/Hillary/Documents/PI/Doc/Stock Screener/mapping_factors.csv')
fundamental=factor_list[factor_list['tables']=='fundamental']
add = pd.DataFrame([['marketcap','fundamental','quarterly']],columns=['variable','tables','frequency'])
fundamental=pd.concat([add,fundamental])
fundamental=fundamental[fundamental.variable !='distrib']

fundamental_merge[list(fundamental['variable'])]=fundamental_merge[list(fundamental['variable'])].apply(pd.to_numeric,errors='coerce')

sum_stat=fundamental_merge[list(fundamental['variable'])].describe()       

period = '_'+start+'_'+end
sum_stat.to_csv("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_summary_statistics"+period+".csv")

#%% Fill missing values

before_fill=pd.DataFrame()
before_fill.to_csv("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_missing_values"+period+".csv")

# mannually copy the .info() results to csv file
fundamental_merge[list(fundamental['variable'])].info()

del fundamental_merge['distrib']

    
def fill_missing(df, var_list, by1, by2, key):
    
    '''
    fill missing value using the following waterfall logic: 
      first by the median of 'by1' group
      then if still missing&by2 is specified, fill by the median of 'by2' group

    '''
    

    df[varlist]=df.groupby(by1)[varlist].transform(lambda x: x.fillna(x.median()))
    

    if len(by2)==0:
          
        return df
       
    elif len(by2)>0:
        
        df[varlist]=df.groupby(by2)[varlist].transform(lambda x: x.fillna(x.median()))
    
        return df

fundamental_merge=fill_missing(fundamental_merge,list(fundamental['variable']),'code',['year','quarter'],['code','year','quarter'])



fundamental_merge.to_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_after_fill_missing"+period+".pkl")

#%% Normalization

fundamental_merge=pd.read_pickle("C:/Users/Hillary/Documents/PI/Code/Working_On/Factor model/fundamental_after_fill_missing"+period+".pkl")

def winsorize_series(s,pctl):
        q = s.quantile([pctl, 1-pctl])
        s[s < q.iloc[0]] = q.iloc[0]
        s[s > q.iloc[0]] = q.iloc[1]
        return s
    
def standardize_series(s,num):
        mean, std = s.mean(), s.std()
        r_outliers = (s - mean) > num*std
        l_outliers = (s - group) > num*std            
        s[r_outliers] = s+num*std
        s[l_outliers] = s-num*std    
        return s
    
def winsroize_df(df, pctl):
    return df.apply(winsorize_series,pctl)
    
    
    
def Normalization(df, method):
    
    '''
    data normalization based on method chosen:
    method: 'win' for winsorization
            'std' for #distance from mean
    
    '''
    if method =='std':
        num = input('enter the number of std from mean: ')
         
             
    elif method =='win':
        pctl = float(input("enter the percentile (in decimal) for winsorization: "))


fundamental_win = fundamental_merge.copy()

def replace(group):
    mean, std = group.mean(), group.std()
    outliers = (group - mean).abs() > 3*std
    group[outliers] = mean        # or "group[~outliers].mean()"
    return group

fundamentals.groupby('').transform(replace)



def identify_outlier(s,distance):
    avg=np.nanmean(s)
    std=np.nanstd(s)
    st=pd.DataFrame(np.where(((s>avg+distance*std)|(s<avg-distance*std)),-999999,s))
    #col_name='ck'+ff
    st.columns=['check']
    chg_pct = st[st.check==-999999].count()/len(st)
    print('Percentage of Outliers is %s'%chg_pct)
    return st
    
def identify_missing(s):
    st=pd.DataFrame(np.where(np.isnan(s),999999,s))
    st.columns=['check']
    chg_pct = st[st.check==999999].count()/len(st)
    print('Percentage of Missing Data is %s'%chg_pct)
    return st
  
#%%


def replace(group):
    mean, std = group.mean(), group.std()
    outliers = (group - mean).abs() > 3*std
    group[outliers] = mean        # or "group[~outliers].mean()"
    return group

fundamentals.groupby('').transform(replace)



#%%


missing_data=pd.DataFrame()
outlier_data=pd.DataFrame()

for ff in ['eps','cf_nm']:
    print('current fundamental factor is',ff)
    stocks[ff]=pd.to_numeric(stocks[ff],errors='coerce')
    try:
        a_missing= stocks.groupby('year')[ff].apply(identify_missing)
        missing_data[ff]=a_missing
    except:
        print('check exception')
        
for ff in ['eps','cf_nm']:
    print('current fundamental factor is',ff)
    stocks[ff]=pd.to_numeric(stocks[ff],errors='coerce')
    try:
        b_outlier=stocks.groupby('year')[[ff]].apply(identify_outlier,distance=3)
        #stocks['check'+'_'+ff]=b_outlier
        outlier_data = 
    except:
        print('check exception')        
        
missing_data.to_csv('C:\Users\Hillary\Documents\PI\Code\Working_On\Factor model\check_missing_cross_section.csv')
        
outlier_data.to_csv('C:\Users\Hillary\Documents\PI\Code\Working_On\Factor model\check_outlier_cross_section.csv')

