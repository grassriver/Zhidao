#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 20:41:50 2018

@author: yunongwu
"""

import pandas as pd
import sqlite3 as sql
import sys
sys.path.append('F:/Nut/PI/Code/')
conn = sql.connect('F:/Nut/PI/data/data.db')
c = conn.cursor()
c.execute("select code from stock_basics")
#c.execute("select distinct code from stocks_price")
#price = pd.DataFrame(c.fetchall())
code_list=pd.DataFrame(c.fetchall())
#code_list.columns=['Code']
#price.columns =['Date','Open','Close','High','Low','Volume','Code']
# a=price.loc[(price['Code']=='000517')|(price['Code']=='600399')]
#b=price.sort_values(by=['Code','Date'])
#b['Date'] = pd.DatetimeIndex(b['Date'])
#b = b.set_index('Date')

#code_list=code_list.Code.unique()
#code_list=pd.DataFrame(code_list)
code_list.columns=['code']
#%%
sys.path.append('F:/Nut/PI/Code/Working_On/')
#import Ratios as ratio
from stock_class import Stock
#import general_tools as tool
#sys.path.append('/Users/yunongwu/Documents/Portfolio_Intelligence/Code/Test/YN_Test')
#from capm import capm_modeling
#from capm import get_date_range
#bus_calender = tool.get_business_calendar(conn)
#mkt_code = 'sh000300'
#stk_list = tool.get_stock_pool(conn,t)
#
#for day in bus_calender['date']:
#    coeff=capm_modeling(conn=conn, t=day,frequency='daily',window=252,mkt_code='sh000001')
#    betas = pd.concat([betas,coeff['code','end','beta']],axis=1)

c.execute("select date, close, code from index_price where code =='sh000001'")
mkt_price = pd.DataFrame(c.fetchall())
mkt_price.columns = ['Date','Close','Code']
start=mkt_price['Date'][0]
end=mkt_price['Date'].iloc[-1]
date_list=mkt_price['Date'][1:]
#date_list2=['2017-12-29','2017-12-28']
returns = Stock(conn,list(code_list),start,end,all_stocks=True).daily_returns

#date_temp = pd.to_datetime(start)
#date_temp = date_temp+pd.offsets.relativedelta(days=1)*252
#%%
sys.path.append('F:/Nut/PI/Code/Test/YN_Test/')
from capm import capm_modeling
#from capm import get_date_range

#coeff=capm_modeling(conn=conn, returns=returns,t='2000-01-05',frequency='daily',window=252,mkt_code='sh000001')

coeff_new=pd.DataFrame()

for t in date_list:   
    coeff=capm_modeling(conn=conn, returns=returns,t=t,frequency='daily',window=252,mkt_code='sh000001')
    coeff['end']=t
    coeff_new=pd.concat([coeff_new,coeff],axis=0)
    
beta = coeff_new[['code','end','beta']]
beta.columns=['code','date','Beta'] 
beta=beta.sort_values(by=['code','date'])
#%%
import time
since = time.time()
c.execute("select * from technical")
technical=pd.DataFrame(c.fetchall())
time_lag = time.time()-since
print('Read technical index table completed in {:.0f}m {:.0f}s'.format(time_lag // 60, time_lag % 60))

technical.columns=['index','code','date','RSI6','RSI12','RSI24','SMA5','SMA10','SMA20','SMA30','SMA60','20HIGH','20LOW','50HIGH','50LOW','GAP','CloseChange','ChangeFromOpen','Amplitude%','High_Range%','Low_Range%','MACD_DIFF','MACD_DEA','MACD']
#technical.head(2)
dfs=[technical,beta]
from functools import reduce
df_final = reduce(lambda left,right: pd.merge(left,right,on=['code','date'],how='left'), dfs)

df_final.to_sql(con=conn, name='technical', if_exists='replace')
conn.commit()

conn.close()
#df_final[['date','Beta']].head(20)
#coeff.to_csv('old.csv')
#t='2017-01-04'
#window=252
#frequency='daily'
#stk_list = tool.get_stock_pool(conn,t)
#stk_list = pd.DataFrame(stk_list)
#stk_list.columns = ['code']
#date_range = get_date_range(conn,t,window,frequency)
#begin = date_range[0]
#begin=begin.strftime('%Y-%m-%d')
#    # get the time period before and including t
#mkt = get_mkt_index(conn,t,window,frequency,mkt_code)[['mkt_ret']]
#    
#data = Stock(conn,list(stk_list['code'][0:800]),begin.strftime('%Y-%m-%d'),t).daily_returns
