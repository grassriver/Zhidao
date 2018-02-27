# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:20:06 2018

@author: Kai Zheng
"""

import pandas as pd
import sqlite3 as sql
from stock_class import Stock
from portfolio_class import Portfolio

conn = sql.connect("d:/Kaizheng/Working_directory/portfolio_intelligence/data/tushare/data.db")

code_list = ['000001', '000002']
s = Stock(conn, code_list)

stocks_list = pd.DataFrame({'code': ['000001','000002'], 'shares': [1000, 1000]})
p = Portfolio(conn, stocks_list)