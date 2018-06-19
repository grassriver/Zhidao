#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:14:26 2018

@author: zifandeng
"""
import pandas as pd
conn_path = conn_path='/Users/zifandeng/Nustore Files/PI/data/'
conn = sql.connect(conn_path+'/data.db')
price_data = pd.read_sql('select a.*,b.timetomarket from stocks_price a left join stock_basics b on a.code = b.code',conn)
price_data['timeToMarket']=list(map(float,price_data['timeToMarket']))
price_data['timeToMarket']= price_data['timeToMarket'].fillna(0)
price_data['timeToMarket']=list(map(int,price_data['timeToMarket']))
price_data['timeToMarket']=list(map(str,price_data['timeToMarket']))
price_data['timeToMarket'][price_data['timeToMarket']=='0'] = 'NaN'
price_data['timeToMarket'] = pd.to_datetime(price_data['timeToMarket'])
price_data['date'] = pd.to_datetime(price_data['date'])
price_data['IPO_flag'] = (price_data['date']>=price_data['timeToMarket'])
price_data.to_pickle('/Users/zifandeng/Nustore Files/PI/Code/Working_On/data checking/prices_with_ipo_date.pickle')

s = price_data['timeToMarket']
initial_date = pd.to_datetime('1900-01-01')
available_list = []
price_data = price_data.sort_values('date')
for d in price_data['date'].unique():
    if d == initial_date:
        available_list=code_list
    else:
        print(d)
        code_list = price_data['code'][s.between(initial_date,d)].unique()
        code_list=list(code_list)
        code_list.insert(0,d)
        available_list.append(code_list)

available_list_long = available_list_pd.melt(id_vars=0,value_name='code')
available_list_long.columns = ['date','index','code']
available_list_long = available_list_long[['date','code']]
available_list_long = available_list_long[~(available_list_long['code']=='600018')]
price_data = price_data[~(price_data['code']=='600018')]
available_list_long = available_list_long[~(available_list_long['code']=='601313')]
price_data = price_data[~(price_data['code']=='601313')]

price_expanded=available_list_long.merge(price_data,how='left',left_on=['date','code'],right_on=['date','code'])
delist = tushare.get_terminated()
delist['tDate'][delist['tDate']=='-']=np.nan
delist['delist_flag'] = 1
delist['tDate']=pd.to_datetime(delist['tDate'])
price_expanded=price_expanded.merge(delist[['code','tDate','delist_flag']],how='left',left_on=['date','code'],right_on=['tDate','code'])

price_expanded['Suspend']=np.isnan(price_expanded['close'])&np.isnan(price_expanded['delist_flag'])
suspend = price_expanded.loc[price_expanded['Suspend']==True,['date','code']]
suspend.to_pickle('/Users/zifandeng/Nustore Files/PI/Code/Working_On/data checking/suspend_list.pickle')

count_info = pd.DataFrame({'停牌数':suspend.groupby('date')['code'].count(),
'交易股票数':price_data.groupby('date')['code'].count(),
'总股票数':available_list_long.groupby('date')['code'].count()})

ipo_count=price_data[['timeToMarket','code']]
ipo_count =ipo_count.drop_duplicates()
ipo_count = ipo_count.groupby('timeToMarket')[['code']].count()
ipo_count.columns = ['新股上市数']
delist_count = delist.groupby('tDate')[['code']].count()
delist_count.columns = ['退市股票数']
count_info_merged = count_info.merge(ipo_count,left_index=True,right_index=True,how='left')
count_info_merged = count_info_merged.merge(delist_count,left_index=True,right_index=True,how='left')
count_info_merged['退市股票数'][np.isnan(count_info_merged['退市股票数'])]=0
count_info_merged['新股上市数'][np.isnan(count_info_merged['新股上市数'])]=0
count_info_merged['总股票数']=count_info_merged['总股票数']-count_info_merged['退市股票数']
count_info_merged.to_excel('/Users/zifandeng/Nustore Files/PI/Code/Working_On/data checking/stock_count.xlsx')

