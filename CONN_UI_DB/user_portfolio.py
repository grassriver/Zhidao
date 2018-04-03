#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:55:56 2018

@author: xichenjin
"""

#%%
import sqlite3 as sql
import pandas as pd

#%%
def create_table_port(conn):
    c=conn.cursor()
    c.execute(
        """create table if not exists
        %s(
        %s integer primary key autoincrement,
        %s varchar(128),
        %s varchar(128),
        %s varchar(128),
        %s varchar(128),
        %s real,
        %s TIMESTAMP)"""
        %('user_portfolio',
          'id',
          'user_name',
          'portfolio_name',
          'industry',
          'stock_code',
          'shares',
          'created_dtm'))
    return c
    
    
#%%
#create portfolio
def create_portfolio(conn,input_name,port_name,ind,stk,wt_s,date):
        c=conn.cursor()
        userdata=c.execute("select * from user where user_name='%s'"%input_name).fetchone()
        if userdata:
            for i in range(len(stk)):                
                c.execute("insert into user_portfolio(user_name, portfolio_name, industry, stock_code, shares, created_dtm) values(?,?,?,?,?,?)", (input_name,port_name,ind,stk[i],wt_s[i],date))
            conn.commit()
            print ("组合建立成功")
        else:
            print ("请先注册")