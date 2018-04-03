# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:54:35 2018

@author: Think
"""
import numpy as np
import pandas as pd
import warnings
#import max_drawdown as md

#%%


class Stock(object):
    """
    parameters:
    -----------
    conn: connection to database
        e.g. import sqlite3 as sql
             conn = sql.connect('datapath/data.db')                 
    code: list
        stock code.
    start: string
        start date.
    end: string
        end date.
    """

    def __init__(self, conn, code, start='2017-01-01', end='2017-12-01'):
        self._code = code
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._conn = conn
        self._price = self.get_data(conn)
        self._industry = self.get_industry(conn)

    def get_data(self, conn):
        if isinstance(self._code, (type, str)):
            query = 'select * from stocks_price where code="%s"' % (self._code)
        elif isinstance(self._code, (type, list)):
            query = 'select * from stocks_price where code in ("%s")' % ('","'.join(self._code))
        else:
            raise ValueError('code can only be a string or a list')
             
        stocks_price = pd.read_sql(query, conn)
        if stocks_price.empty:
            raise ValueError('no data fetched')
        stocks_price['date'] = pd.to_datetime(stocks_price['date'])
        stocks_price.set_index('date', inplace=True)
        self._price = pd.DataFrame()
        for code in self._code:
            temp = stocks_price[stocks_price.code==code]
            temp2 = temp.iloc[(temp.index >= self._start) & (temp.index <= self._end), :]
            if temp2.empty:
                raise ValueError('No data for stock {} in start_to_end period.'.format(code))
            elif self._start != temp2.index.min():
                warnings.warn('Data of stock {} not available for start date'.format(code))
            self._price = self._price.append(temp2)

        return self._price
    
    def get_industry(self,conn):
        query = 'select code,name,industry from stock_basics where code in ("%s")' % ('","'.join(self._code))
        industry = pd.read_sql(query, conn) 
        if industry.empty:
            industry = None
        industry.set_index('code', inplace=True)
        return industry
    
    @property
    def code(self):
        return self._code

    @property
    def industry(self):
        return self._industry

    @property
    def price(self):
        return self._price

    @property
    def close_price(self):
        """
        return a pandas series of close price
        """
        df = pd.DataFrame.assign(self._price)
        self._close_price = df.pivot(columns='code', values='close')
        self._close_price = self._close_price[self._code]
        return self._close_price

    @property
    def daily_returns(self):
        """
        return a pandas series of daily returns
        """
        df = pd.DataFrame.assign(self._price)
        df['return'] = np.log(df['close'] / df.groupby(by='code')['close'].shift(1))
        df.dropna(axis=0, how='any', inplace=True)
        self._daily_returns = df.pivot(columns='code', values='return')
        return self._daily_returns

    @property
    def daily_cum_returns(self):
        """
        return a pandas series of daily cumulative returns
        """
        df = self._price
        daily_cum_returns = np.log(df.pivot(columns='code', values='close') / 
                                   df.groupby(by='code')['close'].first())
        self._daily_cum_returns = daily_cum_returns
        return self._daily_cum_returns
        
#        self._daily_cum_returns = np.log(self._close_price / self._close_price.iloc[0])
#        return self._daily_cum_returns
    
#    def gen_drawdown_table(self, top=10):
#        drawdown_table = md.gen_drawdown_table(self.daily_returns['daily returns'], top)
#        return drawdown_table
#    
#    def plot_drawdown_periods(self, top=10, ax=None, **kwargs):
#        ax = md.plot_drawdown_periods(self.daily_returns['daily returns'], top, ax, **kwargs)
#        return ax
#    
#    def plot_drawdown_underwater(self, ax=None, **kwargs):
#        ax = md.plot_drawdown_underwater(self.daily_returns['daily returns'], ax, **kwargs)
#        return ax
    
        
