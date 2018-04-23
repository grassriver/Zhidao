#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:04:41 2018

@author: zifandeng
"""
import pandas as pd
import numpy as np
import warnings
import math
import statsmodels.api as sm
from stock_class import Stock


def prompt_for_share(code_list, name_list=None):
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
    if name_list is None:
        for i in range(len(code_list)):
            s = int(input('Please enter the share for stock ' + code_list[i] + ': '))
            if((s / 100) != np.floor(s / 100)):
                raise ValueError('Please enter a valid share number')
            share.insert(i, s)
    else:
        for i in range(len(code_list)):
            s = int(input('Please enter the share for stock ' + code_list[i] + ' ' + name_list[i] + ': '))
            if((s / 100) != np.floor(s / 100)):
                raise ValueError('Please enter a valid share number')
            share.insert(i, s)

    return pd.DataFrame({'code': code_list,
                         'shares': share})


def prompt_for_weight(code_list, name_list=None):
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
    if name_list is None:
        for i in range(len(code_list)):
            if(i == (len(code_list) - 1)):
                threshold_low = round(1 - sum(weight), 2)
            else:
                threshold_low = 0
            threshold_high = round(1 - sum(weight), 2)
            w = float(input('Please enter the weight for stock ' + code_list[i] + "(range[%s): " % (str(threshold_low) + ',' + str(threshold_high) + ']')))
            if((w < 0) | (w > 1)):
                raise ValueError('Please enter a valid weight')
            weight.insert(i, w)
    else:
        for i in range(len(code_list)):
            if(i == (len(code_list) - 1)):
                threshold_low = round(1 - sum(weight), 2)
            else:
                threshold_low = 0
            threshold_high = round(1 - sum(weight), 2)
            w = float(input('Please enter the weight for stock ' + code_list[i] + ' ' + name_list[i] + "(range[%s): " % (str(threshold_low) + ',' + str(threshold_high) + ']')))
            if((w < 0) | (w > 1)):
                raise ValueError('Please enter a valid weight')
            weight.insert(i, w)

    if(sum(weight) != 1):
        raise ValueError('Sum of the weight not equals to 1, Please enter the valid weights')
    return pd.DataFrame({'code': code_list,
                         'weight': weight})

# 1. check code list


def check_code_list(conn, code_list, start):
    c = conn.cursor()
    c.execute("select code from stocks_price where date = " + "'" + start + "' and code in ('%s')" % ("','".join(code_list)))
    db = pd.DataFrame(c.fetchall())
    if(db.empty):
        raise ValueError('No data available for selected date')
    code = db.iloc[:, 0]
    code_list = pd.DataFrame(code_list).iloc[:, 0]

    if(not(code_list.isin(code).all())):
        code_ex = code_list[~(code_list.isin(code))]
        warnings.warn('Data of stock %s are not available for start date' % ((",".join(code_ex))))
    return code

# 2. equal weight


def weight_calc(conn, code_list, name_list, start, capital, equal=False):
    c = conn.cursor()
    c.execute("select code,close from stocks_price where date = " + "'" + start + "' and code in ('%s')" % ("','".join(code_list)))
    price = pd.DataFrame(c.fetchall())
    price.columns = ['code', 'price']
    if(equal == True):
        price['weight'] = np.repeat(1 / len(code_list), len(code_list))
    else:
        price['weight'] = prompt_for_weight(code_list, name_list).weight
    price['capital'] = np.repeat(capital, len(code_list))
    price['capital_per_stock'] = price['weight'] * price['capital']
    price['pre_share'] = price['capital_per_stock'] / price['price']
    price['shares'] = np.floor(price['pre_share'] / 100) * 100
    return price[['code', 'shares']]


def portfolio_construct(conn, start, code_list, name_list=None,
                        capital=1000000, construct_type='share', equal=False):
    code_list = check_code_list(conn, code_list, start)
    if(construct_type == 'share'):
        if(equal == False):
            db = prompt_for_share(code_list, name_list)
        else:
            s = int(input('Please enter the share: '))
            share = np.repeat(s, len(code_list))
            db = pd.DataFrame({'code': code_list,
                               'shares': share})
    else:
        #capital = float(input('Please enter the total capital: '))
        if(equal == False):
            db = weight_calc(conn, code_list, name_list, start, capital, False)
        else:
            db = weight_calc(conn, code_list, name_list, start, capital, True)
    return db


def get_lagged_time(date, freq, lag):
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
        year = (month <= 0) * (year + math.floor((month - 0.25) / 12)) + (month > 0) * year
        month = (month <= 0) * ((month) % -12 + 12) + (month > 0) * month
        date2 = str(year * 100 + month)
        date2 = date2[0:4] + '-' + date2[4:6]

    if(freq == 'quarterly'):
        quarter = (math.ceil(int(month) / 3)) - lag
        year = (quarter <= 0) * (year + math.floor((quarter - 0.25) / 4)) + (quarter > 0) * year
        quarter = (quarter <= 0) * ((quarter) % -4 + 4) + (quarter > 0) * quarter
        date2 = str(year) + 'Q' + str(quarter)

    if(freq == 'daily'):
        date = pd.to_datetime(date)
        off1 = pd.DateOffset(lag)
        date = date - off1
        date2 = str(date.year * 10000 + date.month * 100 + date.day)
        date2 = date2[0:4] + '-' + date2[4:6] + '-' + date2[6:8]

    if(freq == 'annually'):
        date2 = str(year - lag)

    return date2


def portfolio_construct_by_weight(conn, start, code_list, weights=None, name_list=None,
                                  capital=1000000, backfill=True,
                                  stocks_price_old=None,
                                  business_calendar=None,
                                  industry=None):
    if weights is None:
        weights = np.array(len(code_list)*[1/len(code_list)])
    else:
        weights = np.array(weights)
    stocks = Stock(conn, list(code_list), start=start, end=start, 
                   backfill=backfill,
                   stocks_price_old=stocks_price_old,
                   business_calendar=business_calendar,
                   industry=industry)
    price = pd.melt(stocks.close_price, value_name='price')
    shares = np.floor(weights*capital/price['price']/100)*100 
    db = pd.DataFrame({'code': price['code'], 'shares': shares})
    return db


def weight2share(conn, code_list, name_list, start, capital, weights):
    c = conn.cursor()
    c.execute("select code,close from stocks_price where date = " + "'" + start + "' and code in ('%s')" % ("','".join(code_list)))
    price = pd.DataFrame(c.fetchall())
    price.columns = ['code', 'price']
    if weights is None:
        price['weight'] = np.repeat(1 / len(code_list), len(code_list))
    else:
        price['weight'] = weights
    price['capital'] = np.repeat(capital, len(code_list))
    price['capital_per_stock'] = price['weight'] * price['capital']
    price['pre_share'] = price['capital_per_stock'] / price['price']
    price['shares'] = np.floor(price['pre_share'] / 100) * 100
    return price[['code', 'shares']]


def get_business_calendar(conn, index='sh000001'):
    query = 'select date from index_price where code = "' + index + '"'
    business_calendar = pd.read_sql(query, conn)
    business_calendar = business_calendar.sort_values('date')
    return business_calendar

def get_industry(conn, code_list):
    query = 'select code,name,industry_zx from stock_basics where code in ("%s")' % ('","'.join(code_list))
    industry = pd.read_sql(query, conn)
    if industry.empty:
        industry = None
        return industry
    industry.set_index('code', inplace=True)
    return industry

def get_stocks_price_old(conn, code_list):
    if isinstance(code_list, (type, list)):
        query = 'select * from stocks_price where code in ("%s")' % ('","'.join(code_list))
    else:
        raise ValueError('code can only be a list')
    stocks_price = pd.read_sql(query, conn)
    return stocks_price

def get_benchmark_old(conn, index_code='sh000001'):
    query = 'select * from index_price where code="%s"' % (index_code)
    index = pd.read_sql(query, conn)
    return index    

def first_trading_day(conn, start, business_calendar=None):
    if business_calendar is None:
        business_calendar = get_business_calendar(conn)
    if start not in business_calendar['date'].tolist():        
        if (start > business_calendar['date'].max() or
            start < business_calendar['date'].min()):
            raise ValueError('start is out of the business calendar range')        
        start = business_calendar[business_calendar['date'] >= start].iloc[0, 0]
        warnings.warn('start date is not a business day, have changed to the next trading day')
    return start

def last_trading_day(conn, start, business_calendar=None):
    if business_calendar is None:
        business_calendar = get_business_calendar(conn)
    if start not in business_calendar['date'].tolist():        
        if (start > business_calendar['date'].max() or
            start < business_calendar['date'].min()):
            raise ValueError('start is out of the business calendar range')        
        start = business_calendar[business_calendar['date'] <= start].iloc[-1, 0]
        warnings.warn('start date is not a business day, have changed to the next trading day')
    return start

def get_stock_pool(conn,t):
    code_list = pd.read_sql_query("select distinct code from stocks_price where date = '"+t+"'",conn)
    return code_list

def autoregression_test(tseries,threshold=0.05):
    [stats,pvalue] = sm.stats.acorr_ljungbox(tseries,20)
    # return if fail to reject the null that no serial correlation
    return (pvalue<threshold).any()

def garch_effect_test(tseries,threshold = 0.05):
    [stats,pvalue] = sm.stats.acorr_ljungbox(tseries*tseries,20)
    # return if fail to reject the null that no serial correlation
    return (pvalue<threshold).any()

def stationary_test(tseries,threshold = 0.05):
    pvalue = sm.tsa.adfuller(tseries)[1]
    #return if reject the null that there is the unit root
    return pvalue<threshold