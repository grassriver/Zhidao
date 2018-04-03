#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:24:57 2018

@author: xichenjin
"""
#%%change working directory
import os
os.chdir('/Users/xichenjin/Desktop/portfolio_intelligence/Working_on/CONN_UI_DB')
#%%
import sqlite3 as sql
import user_info as uinfo
import user_portfolio as uport
import numpy as np
import pandas as pd
#%%
conn=sql.connect('/Users/xichenjin/Desktop/portfolio_intelligence/Working_on/CONN_UI_DB/user_data.db')
input_name="sherry"
input_password="Jxc0126"
#%%create table
uinfo.create_sql(conn)
uport.create_table_port(conn)
#%%registration
uinfo.register(conn,input_name, input_password)
#%%log in
uinfo.val(conn, input_name, input_password)
print ("进入用户操作")
#%%create portfolio
uport.create_portfolio(conn,input_name,"makemoney","finance","000001",500,"2017-09-10 12:00:00")