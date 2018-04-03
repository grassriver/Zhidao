# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:44:06 2018

@author: Kai Zheng
"""

import pandas as pd
import sqlite3 as sql
from functools import reduce

conn = sql.connect("d:/Kaizheng/Working_directory/portfolio_intelligence/data/tushare/data2.db")
c = conn.cursor()

c.execute('select * from report_data')
report_data = pd.DataFrame(c.fetchall(), columns=['code', 'name', 'eps', 'eps_yoy', 'bvps',
                                                  'roe', 'epcf', 'net_profits', 'profits_yoy',
                                                  'distrib', 'report_date', 'date'])
report_data2 = report_data[['code', 'name', 'eps_yoy', 'bvps',
                            'epcf', 'profits_yoy',
                            'distrib', 'report_date', 'date']]

c.execute('select * from profit_data')
profit_data = pd.DataFrame(c.fetchall(), columns=['code', 'name', 'roe', 'net_profit_ratio',
                                                  'gross_profit_rate', 'net_profits', 'eps',
                                                  'business_income', 'bips', 'date'])

c.execute('select * from operation_data')
operation_data = pd.DataFrame(c.fetchall(), columns=['code', 'name', 'arturnover', 'arturndays',
                                                                                   'inventory_turnover', 'inventory_days',
                                                                                   'currentasset_turnover', 'currentasset_days',
                                                                                   'date'])

c.execute('select * from growth_data')
growth_data = pd.DataFrame(c.fetchall(), columns=['code', 'name', 'mbrg', 'nprg', 'nav', 'targ',
                                                  'epsg', 'seg', 'date'])

c.execute('select * from debtpaying_data')
debtpaying_data = pd.DataFrame(c.fetchall(), columns=['code', 'name', 'currentratio', 'quickratio',
                                                      'cashratio', 'icratio', 'sheqratio',
                                                      'adratio', 'date'])

c.execute('select * from cashflow_data')
cashflow_data = pd.DataFrame(c.fetchall(), columns=['code', 'name', 'cf_sales', 'rateofreturn',
                                                    'cf_nm', 'cf_liabilities', 'cashflowratio',
                                                    'date'])

dfs = [report_data2, profit_data, operation_data, growth_data, debtpaying_data, cashflow_data]
fundamental = reduce(lambda left, right: pd.merge(left, right, on=['code', 'name', 'date'], how='outer'), dfs)

fundamental.to_sql(con=conn, name='fundamental', if_exists='replace')
conn.commit()

conn.close()
