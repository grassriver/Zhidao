#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:39:59 2018

@author: xichenjin
"""
import sqlite3 as sql
#%%
#create a database
def create_sql(conn):
    c=conn.cursor()
    c.execute(
        """create table if not exists
        %s(
        %s integer primary key autoincrement,
        %s varchar(128),
        %s varchar(128))"""
        %('user',
          'id',
          'user_name',
          'password'))
#   conn.close()
    return c
#%%
#register
def register(conn,input_name,input_password):
        c=conn.cursor()
        userdata=c.execute("select * from user where user_name='%s'"%input_name).fetchone()
        if not userdata:
            c.execute("insert into user(user_name, password) values(?,?)", (input_name, input_password))
            conn.commit()
            print ("注册成功")
#           conn.close()
#           break
        else:
            print ("用户已存在")
#%%
#log in
def val(conn,input_name,input_password):
        data=showdata(conn, input_name) #retrieve user info
        if data:
            if data[2]==input_password:
                print ("登陆成功")
            else:
                print ("密码错误")
        else:
            print ("用户名错误")
#%%
#read data from user_data
def showdata(conn, username):
    c=conn.cursor()
    userdata=c.execute("select * from user where user_name='%s'"%username).fetchone()
#   conn.close()
    return userdata
#%%