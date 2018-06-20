# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 22:13:11 2018
Last update Mar 17 2018

@author: H.Wang
"""

#%%
import sqlite3 as sql
import pandas as pd
import numpy as np
import sys
import datetime
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num

conn_path='C:/Users/Hillary/Documents/PI/data/data.db'

 
def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
    
    """
    :param dat: pandas DataFrame object with datetime64 index, 
                and float columns "open", "high", "low", and "close"
    :param stick: A string or number indicating the period of time covered by a single candlestick. 
                  Valid string inputs include "day", "week", "month", and "year", ("day" default)
                 
    :param otherseries: An iterable that will be coerced into a list, 
                 containing the columns of dat that hold other series to be plotted as lines
 
    """
    
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # Make ticks on occurrences of each day of the month,minor ticks
    dayFormatter = DateFormatter('%d')      # e.g., 12
    
    transdat = dat.loc[:,["open", "high", "low", "close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting, defaul to 1 for day-plot
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] =     pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"open": [], "high": [], "low": [], "close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"open": group.iloc[0,0],
                                            "high": max(group.high),
                                            "low": min(group.low),
                                            "close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
 
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('365 days'):
        weekFormatter = DateFormatter('%b %d, %Y')  # e.g., Jan 12,2007
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_major_formatter(weekFormatter)
    else:
        
        weekFormatter = DateFormatter('%b %d, %Y')
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
        ax.xasix.set_major_formatter(weekFormatter)
       
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), 
                      plotdat["open"].tolist(), plotdat["high"].tolist(),
                      plotdat["low"].tolist(), plotdat["close"].tolist())),
                      colorup = "red", colordown = "green", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
 
    plt.show()
    return fig


# test the code with data pulled from database
    
#%%
def stock_plot(conn,stick,stk_code,start,end,see):
    
    
    c=conn.cursor()
    
    
    if isinstance(stk_code,(type,str)):
     c.execute('select * from stocks_price where code="%s"' % (stk_code))
    elif isinstance(stk_code,(type,list)):
     c.execute('select * from stocks_price where code in ("%s")' % ('","'.join(stk_code)))

    stk = pd.DataFrame(c.fetchall())
    if stk.empty:
            raise ValueError('chosen stock not found in the database')
            
    stk.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'code']
    stk['date']=pd.to_datetime(stk['date'])
    stk.set_index('date',inplace=True)
    
    print('Data is availabe from %s to %s for the chosen stock'%(stk.index.min(),stk.index.max()))
    if pd.to_datetime(start)<stk.index.min() or pd.to_datetime(end)>stk.index.max():
           raise ValueError ("Selected period is out of bound")  

    print('%sly'%(stick),'Plots for stock %s from %s to %s'%(stk_code,start,end))
    
    stk=stk.sort_index(axis=0)
    
    if stick=='day':
          stk["5day"]=np.round(stk["close"].rolling(window=5,min_periods=1,center=False).mean(),2)
          stk["20day"]=np.round(stk["close"].rolling(window=20,min_periods=1,center=False).mean(),2)
          stk["60day"]=np.round(stk["close"].rolling(window=60,min_periods=1,center=False).mean(),2)
          fig=pandas_candlestick_ohlc(stk.loc[start:end,:],stick = stick,otherseries=["5day","20day","60day"])
    elif stick=='week':
          stk["5week"]=np.round(stk["close"].rolling(window=35,min_periods=1,center=False).mean(),2)
          stk["20week"]=np.round(stk["close"].rolling(window=140,min_periods=1,center=False).mean(),2)
          stk["60week"]=np.round(stk["close"].rolling(window=420,min_periods=1,center=False).mean(),2)
          fig=pandas_candlestick_ohlc(stk.loc[start:end,:],stick = stick,otherseries=["5week","20week","60week"])
    elif stick=='month':   
          stk["6mon"]=np.round(stk["close"].rolling(window=180,min_periods=1,center=False).mean(),2)
          stk["1yr"]=np.round(stk["close"].rolling(window=365,min_periods=1,center=False).mean(),2)
          stk["3yr"]=np.round(stk["close"].rolling(window=1095,min_periods=1,center=False).mean(),2)
          fig=pandas_candlestick_ohlc(stk.loc[start:end,:],stick = stick,otherseries=["6mon","1yr","3yr"])
    else:
          fig=pandas_candlestick_ohlc(stk.loc[start:end,:],stick = stick,otherseries=None)
          print('trends are not provided for yearly plots')
          
 
    date_range = pd.date_range(start=start,end=end,freq='D')
    calender_days=pd.DataFrame({'weekday':np.arange(1,1+len(date_range)),'date':date_range})
    calender_days['date']=pd.to_datetime(calender_days['date'])
    calender_days.set_index('date')
    #isoweekday(), Monday is 1 and Sunday is 7
    #for i in range(len(date_range)):
         #calender_days['weekday'][i]=calender_days['date'][i].isoweekday()
    calender_days['weekday']=calender_days['date'].apply(lambda x: x.isoweekday())

    trading_days=calender_days[calender_days.weekday<6] 
    trading_days.loc['date']=pd.to_datetime(trading_days['date'])
    trading_days.set_index('date',inplace=True)
    user_date = stk.loc[start:end,:]
    
    if len(user_date)<len(trading_days):
        print('there are %s days with no tradings'%(len(trading_days)-len(user_date)))
        if see=='y':
            check_data=trading_days.join(user_date).fillna(-9999)
            no_trading_days = check_data[(check_data['close']==-9999)]
            print('the no trading days are %s'%list(no_trading_days.index))
        else:
            no_trading_days = []
            print('close')


    return fig, no_trading_days

#%%
def prompt():
    stick=input('select plot frequency by entering one of the following: day, week, month, year: ')
    stk_code = input('Please enter your stock code: ')
    start=input('please enter starting point: ')
    end=input('please enter ending point: ')
    see=input('would you like to see the non-trading days? y/n: ')
    
    return stick,stk_code,start,end,see
    
#[stick,stk_code,start,end,see] = prompt()

#[fig, no_traing_days]=stock_plot(conn_path,stick,stk_code,start,end,see)
