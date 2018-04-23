#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:12:13 2018

@author: zifandeng
"""
import numpy as np
import pandas as pd
import sqlite3 as sql
import math

def date_freq_transfer(date,freq):
    """
    Transfer the date input information to corresponding frequency
    
    Parameters
    ----------
    date:
        string; the character string in the format of 'yyyy-mm-dd'
    freq: 
        string; the character string of 'annually','quarterly','monthly','daily'
    
    Returns
    -------
    date2 : string
        The transformed date format
    """
    year = (date[0:4])
    month = (date[5:7])
    day = (date[8:10])
    if(freq == 'monthly'):
        date2 = str(year)+'-'+str(month)
    if(freq == 'quarterly'):
        quarter = (math.ceil(int(month)/3))
        date2 = str(year)+'Q'+str(quarter)
    if(freq == 'daily'):
        date2 = str(year)+'-'+str(month)+'-'+str(day)
    if(freq == 'annually'):
        date2 = str(year)
    return date2

def table_lookup(conn,var):
    
    """
    Look up the table in database for input stock factor
    
    Parameters
    ----------
    conn:
        sqlite 3 connection
    var: 
        string; the character string of input stock factor
    
    Returns
    -------
    date2 : string
        The transformed date format
    """
    c = conn.cursor()
    c.execute("select frequency,tables from mapping where variable = '"+var+"'")
    info = (c.fetchall())
    if len(info)==0:
        raise ValueError('No data for inputs variables!')
    else:
        return info[0][0],info[0][1]

def get_data(conn,var,date,table_name,order='ascending',condition ='None',threshold = 'None'):
    
    """ 
    get data from database
    
    Parameters
    ----------
    conn:
        sqlite 3 connection
   
    var: 
        string; the character string of input stock factor
    
    date:
        string; the character string of input date in the format of 'yyyy-mm-dd'
    
    table_name:
        string; the character string of corresponding table in database
    
    order:
        string; 'ascending'/'descending'
    
    condition:
        string; character for conditions. i.e.,'>=','<=','>','<'
    
    threshold:
        float;the threshold for conditions, can be any floating number
        
    
    Returns
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """    
    
    c = conn.cursor()
    c.execute("pragma table_info('"+table_name+"')")
    data_info = pd.DataFrame(c.fetchall())
    data_list = data_info.iloc[:,1]   
    if(order =='ascending'):
        order = 'asc'
    else:
        order = 'desc'
    if(condition == 'None'):
        c.execute("select code,date, "+var+" from "+table_name+" where date = "+"'"+date+"' and "+var+" is not NULL"+" order by "+var+" "+order)
        db = pd.DataFrame(c.fetchall())
        if(db.empty):
            raise ValueError('No Data Fetched!')
        db.columns = ['Code','Time',var]
        return db
    elif(var in list(data_list)):
        c.execute("select code,date, "+var+" from "+table_name+" where date = "+"'"+date+"' and "+var+" "+condition+str(threshold)+" order by "+var+" "+order)
        db = pd.DataFrame(c.fetchall())
        if(db.empty):
            raise ValueError('No Data Fetched!')
        db.columns = ['Code','Time',var]
        return db
    else:
        return np.nan

def select_top(conn_path,var,date,industry = 'None',since_ipo = {'condition': '>=', 't': 0},top = 30,order='ascending'):
       
    """ 
    select the first/last several stocks for user input stock factor
    
    Parameters
    ----------
    conn:
        sqlite 3 connection
   
    var: 
        string; the character string of input stock factors
    
    date:
        string; the character string of input date in the format of 'yyyy-mm-dd'
        
    order:
        string; 'ascending'/'descending'
    
    top:
        int; the user input integer number to get first/last n stocks with default n = 30

    Returns
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """  
    conn = sql.connect(conn_path+'/data.db')          
    freq,table_name = table_lookup(conn,var)
    date = date_freq_transfer(date,freq)
    db = get_data(conn,var,date,table_name,order = order)
    db = (db.drop_duplicates())
    industry_table = pd.read_excel(conn_path+'/Industry.xlsx',dtype=str)
    db = pd.merge(db,industry_table,how = 'left',left_on = 'Code',right_on='Code')    
    if industry == 'None':
        db = db.iloc[range(min(top,len(db)))]
        db[var+' rank in universe'] = range(1,len(db)+1)
    else:
        if isinstance(industry,str):
            db = db[db['Industry']==(industry)]
        else:
            db = db[db['Industry'].isin(industry)] 
        db = db.iloc[range(min(top,len(db)))]
        db[var+' rank in selected industries'] = range(1,len(db)+1)
    return db

def stock_screener_filter_condition(conn_path,var_list,date,condition_list,threshold_list,industry='None',view_all = True,top = 30):
    
    """  
    select stocks which meet the criteria for user input stock factors
    
    The criteria should only be some filter conditions for value of factor
    
    e.g. user choose to select the stock which santisfies the net_profit>=100000;
    
    Parameters
    ----------
    conn_path:
        string; the connection path for database
   
    var_list: 
        list of strings; the list of character strings of input stock factors
    
    date:
        string; the character string of input date in the format of 'yyyy-mm-dd'

    condition_list:
        list of strings; the list of character string for conditions. i.e.,'>=','<=','>','<'
    
    threshold_list:
        list of floats;the list of float for threshold of conditions, can be any floating number
            
    view_all:
        boolean; default true, the selection to choose to return the entire selected stock data or not
    
    top:
        int; the user input integer number to get first/last n stocks with default n = 30.
        Note that this argument would only be valid if and only if view_all = False
        
    
    Returns
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """       
    conn = sql.connect(conn_path+'/data.db')          
    freq,table_name = table_lookup(conn,var_list[0])
    date = date_freq_transfer(date,freq)
    db = get_data(conn,var_list[0],date,table_name,condition = condition_list[0],threshold = threshold_list[0])
    db = (db.drop_duplicates())
    n = 1
    while(n<len(var_list)):
        freq,table_name = table_lookup(conn,var_list[n])
        date = date_freq_transfer(date,freq)
        temp = get_data(conn,var_list[n],date,table_name,condition = condition_list[n],threshold = threshold_list[n])
        temp = (temp.drop_duplicates())
        db = db.merge(pd.DataFrame(temp[['Code',var_list[n]]]),how = 'inner',left_on = 'Code',right_on = 'Code')
        n = n + 1
    if(db.empty):
        raise ValueError('No Stock meets criteria!')
    industry_table = pd.read_excel(conn_path+'/Industry.xlsx',dtype=str)
    db = pd.merge(db,industry_table,how = 'left',left_on = 'Code',right_on='Code')
    if industry == 'None':
        db = db
    else:
        if isinstance(industry,str):
            db = db[db['Industry']==(industry)]
        else:
            db = db[db['Industry'].isin(industry)]
    if(view_all):
        return db
    else:
        db = db.iloc[range(min(top,len(db))),:]
        return db

def stock_screener_filter_top(conn_path,var_list,date,order,top,industry='None',since_ipo = {'condition': '>=', 't': 2},in_universe = False):
    """  
    select several stocks which meet the creteria for user input stock factors
    
    The creteria should only be some ranking conditions for value of factor
    
    e.g. user would like to choose to select the stock which has the top 30 net_profit 
    and top 10 roe over all stocks as of that date 
    
    Parameters
    ----------
    conn_path:
        string; the connection path for database
   
    var_list: 
        list of strings; the list of character strings of input stock factors
    
    date:
        string; the character string of input date in the format of 'yyyy-mm-dd'
    
    order:
        list of strings; a list of 'ascending'/'descending'

    top:
        list of int; the list of user input integer number to get first/last n stocks.
                
    Returns
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """      
    if in_universe == True:
        industry2 = 'None'
    else:
        industry2 = industry
    db = select_top(conn_path,var_list[0],date,industry = industry2,top = top[0],order = order[0])
    n = 1
    while(n<len(var_list)):
        temp = select_top(conn_path,var_list[n],date,industry = industry2,top=top[n],order = order[n])
        db = db.merge(pd.DataFrame(temp.iloc[:,[0,2,4]]),how = 'inner',left_on = 'Code',right_on = 'Code')
        n = n + 1
    
    if industry == 'None':
        db = db
    else:
        if isinstance(industry,str):
            db = db[db['Industry']==(industry)]
        else:
            db = db[db['Industry'].isin(industry)]
    if(db.empty):
        raise ValueError('No Stock meets criteria!')
    return db



def stock_screener_ranking(conn_path,var_list,date,rank_by,industry='None',since_ipo = {'condition': '>=', 't': 2},in_universe=False,top=50,order='ascending'):
    
    """  
    select stocks as well as user input stock factors with rank on one primary factor
        
    e.g. user would like to see the roe,net_profit,and capital ratio for stocks but
    with net_profit ranked in top 30 over all stocks at that time
        
    Parameters
    ----------
    conn_path:
        string; the connection path for database
   
    var_list: 
        list of strings; the list of character strings of input stock factors
    
    date:
        string; the character string of input date in the format of 'yyyy-mm-dd'
    
    rank_by:
        string; the primariy key factor for ranking in var_list
    
    order:
        strings; a character string of 'ascending'/'descending'

    top:
        int; the input integer number to get first/last n stocks.
                
    Returns
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """    
    if in_universe == True:
        industry2 = 'None'
    else:
        industry2 = industry
    conn = sql.connect(conn_path+'/data.db')          
    var_list.remove(rank_by)
    var_list.insert(0,rank_by)
    db = select_top(conn_path,var_list[0],date,industry = industry2,top=top,order = order)
    n = 1
    while(n<len(var_list)):
        freq,table_name = table_lookup(conn,var_list[n])
        date = date_freq_transfer(date,freq)
        temp = get_data(conn,var_list[n],date,table_name)
        temp = (temp.drop_duplicates())
        db = db.merge(pd.DataFrame(temp[['Code',var_list[n]]]),how = 'left',left_on = 'Code',right_on = 'Code')
        n = n + 1
    if industry == 'None':
        db = db
    else:
        db = db[db['Industry'].isin(list(industry))]
    if(db.empty):
        raise ValueError('No Stock meets criteria!')
    return db


def get_report_date(conn,code_list,date):
    
    """  
    Get the financial report release date
    
    Parameters
    ----------
    conn:
        sqlite 3 connection
    code_list:
        list of string; A list of character string
    date:
        string; A string for quarter
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """      
    
    c = conn.cursor()
    c.execute("select code,name,report_date from fundamental where date = "+"'"+date+"' and code in ('%s')" % ("','".join(code_list)))
    db = pd.DataFrame(c.fetchall())
    if(db.empty):
        raise ValueError('No Report Date Available!')
    return db     

def print_data(conn_path,db):
    
    """  
    Print Data
    
    Parameters
    ----------
    conn_path:
        database connection path
    db:
        the database which needs to be printed
    -------
    db : pandas dataframe
        the database which needs to be printed
    """      
    
    conn = sql.connect(conn_path+'/data.db')          
    db_date = get_report_date(conn,list(db.Code),pd.unique(db.Time)[0])
    db_date.columns = ['Code','Name','Report Release Date']
    db = db_date.merge(db,how = 'left',left_on = 'Code',right_on = 'Code')
    return db
