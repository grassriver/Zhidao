# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:38:11 2018

@author: Kai Zheng
"""

import pandas as pd
import sqlite3 as sql

conn = sql.connect('d:/Kaizheng/Working_directory/portfolio_intelligence/PI/Data/data2.db')
mapping = pd.read_csv('d:/Kaizheng/Working_directory/portfolio_intelligence/PI/Doc/Stock Screener/mapping.csv')
mapping.to_sql(con=conn, name='mapping', if_exists='replace', index=False)
conn.commit()
conn.close()
