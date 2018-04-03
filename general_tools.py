#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:04:41 2018

@author: zifandeng
"""
import pandas as pd
import numpy as np
import warnings
import sqlite3 as sql
import math

def prompt_for_share(code_list,name_list = None):    
    """  
    prompt the argument for user to enter the share given a code list
    
    Parameters
    ----------
    code_list:
        list of string; A list of character string
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """        
    share = []
    if((name_list== None).any()):
        for i in range(len(code_list)):
            s = int(input('Please enter the share for stock '+code_list[i]+': '))
            if((s/100)!=np.floor(s/100)):
                raise ValueError('Please enter a valid share number')
            share.insert(i,s) 
    else:
       for i in range(len(code_list)):
           s = int(input('Please enter the share for stock '+code_list[i]+' '+name_list[i]+': '))
           if((s/100)!=np.floor(s/100)):
                raise ValueError('Please enter a valid share number')
           share.insert(i,s)   
           
    return pd.DataFrame({'code':code_list,
                         'shares':share})

def prompt_for_weight(code_list,name_list = None):    
    """  
    prompt the argument for user to enter the share given a code list
    
    Parameters
    ----------
    code_list:
        list of string; A list of character string
    -------
    db : pandas dataframe
        The dataframe for queried stock information
    """        
    weight = []
    if((name_list== None).any()):
        for i in range(len(code_list)):
            if(i==(len(code_list)-1)):
                threshold_low = round(1- sum(weight),2)
            else:
                threshold_low = 0
            threshold_high = round(1- sum(weight),2)
            w = float(input('Please enter the weight for stock '+code_list[i]+"(range[%s): " % (str(threshold_low)+','+str(threshold_high)+']')))
            if((w<0)|(w>1)):
                raise ValueError('Please enter a valid weight')
            weight.insert(i,w) 
    else:
       for i in range(len(code_list)):
           if(i==(len(code_list)-1)):
                threshold_low = round(1- sum(weight),2)
           else:
                threshold_low = 0
           threshold_high = round(1- sum(weight),2)            
           w = float(input('Please enter the weight for stock '+code_list[i]+' '+name_list[i]+"(range[%s): " % (str(threshold_low)+','+str(threshold_high)+']')))
           if((w<0)|(w>1)):
                raise ValueError('Please enter a valid weight')
           weight.insert(i,w)
           
    if(sum(weight)!=1):
        raise ValueError('Sum of the weight not equals to 1, Please enter the valid weights')             
    return pd.DataFrame({'code':code_list,
                         'weight':weight})
    
# 1. check code list
def check_code_list(conn,code_list,start):
    c = conn.cursor()
    c.execute("select code from stocks_price where date = "+"'"+start+"' and code in ('%s')" % ("','".join(code_list)))
    db = pd.DataFrame(c.fetchall())
    if(db.empty):
        raise ValueError('No data available for selected date')   
    code =db.iloc[:,0]
    code_list = pd.DataFrame(code_list).iloc[:,0]
    
    if(not(code_list.isin(code).all())):
        code_ex = code_list[~(code_list.isin(code))]
        warnings.warn('Data of stock %s are not available for start date'%((",".join(code_ex))))
    return code

# 2. equal weight
def weight_calc(conn,code_list,name_list,start,capital,equal=False):
    c = conn.cursor()
    c.execute("select code,close from stocks_price where date = "+"'"+start+"' and code in ('%s')" % ("','".join(code_list)))
    price = pd.DataFrame(c.fetchall())
    price.columns = ['code','price']
    if(equal == True):
        price['weight'] =  np.repeat(1/len(code_list),len(code_list))
    else:
        price['weight']=prompt_for_weight(code_list,name_list).weight
    price['capital'] = np.repeat(capital,len(code_list))
    price['capital_per_stock'] = price['weight']*price['capital']
    price['pre_share'] = price['capital_per_stock']/price['price']
    price['shares'] = np.floor(price['pre_share']/100)*100
    return price[['code','shares']]
    
def portfolio_construct(conn,start,code_list,name_list = None,capital=1000000,construct_type='share',equal = False):
    code_list = check_code_list(conn,code_list,start)
    if(construct_type == 'share'):
        if(equal == False):
            db = prompt_for_share(code_list,name_list)
        else:
            s = int(input('Please enter the share: '))
            share = np.repeat(s,len(code_list))
            db = pd.DataFrame({'code':code_list,
                               'shares':share})
    else:
        #capital = float(input('Please enter the total capital: '))
        if(equal == False):
            db = weight_calc(conn,code_list,name_list,start,capital,False)
        else:
            db = weight_calc(conn,code_list,name_list,start,capital,True)
    return db


def get_lagged_time(date,freq,lag):
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
    year = int(date[0:4])
    month = int(date[5:7])
    
    if(freq == 'monthly'):
        month = month - lag
        year  = (month<=0)*(year + math.floor((month-0.25)/12))+(month>0)*year
        month = (month<=0)*((month)%-12+12)+(month>0)*month
        date2 = str(year*100+month)
        date2 = date2[0:4]+'-'+date2[4:6]
        
    if(freq == 'quarterly'):
        quarter = (math.ceil(int(month)/3))-lag
        year  = (quarter<=0)*(year + math.floor((quarter-0.25)/4))+(quarter>0)*year
        quarter = (quarter<=0)*((quarter)%-4+4)+(quarter>0)*quarter
        date2 = str(year)+'Q'+str(quarter)
        
    if(freq == 'daily'):
        date = pd.to_datetime(date)
        off1 =  pd.DateOffset(lag)
        date = date-off1
        date2 = str(date.year*10000+date.month*100+date.day)
        date2 = date2[0:4]+'-'+date2[4:6]+'-'+date2[6:8]
        
    if(freq == 'annually'):
        date2 = str(year-lag)
    
    return date2