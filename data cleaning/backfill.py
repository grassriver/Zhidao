# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:21:12 2018

@author: Kai Zheng
"""

import pandas as pd
import sqlite3 as sql

conn = sql.connect('d:/Kaizheng/Working_directory/portfolio_intelligence/PI/data/data.db')

query1 = 'select * from stocks_price'
stocks_price = pd.read_sql(query1, conn)

query2 = 'select date from index_price where code="sh000001"'
date = pd.read_sql(query2, conn)

stocks_price2 = pd.merge(stocks_price, date, how='right', on='date')
stocks_price2 = stocks_price2.sort_values(by = ['code', 'date'])
stocks_price2['date'] = pd.to_datetime(stocks_price2['date'])
stocks_price2 = stocks_price2.set_index('date')
stocks_price2 = stocks_price2.groupby('code').ffill()

stocks_price2.to_sql(name='backfilled_price', con=conn, if_exists='replace')
