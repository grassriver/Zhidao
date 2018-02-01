# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:54:35 2018

@author: Think
"""
import numpy as np
import pandas as pd

#%%
class Stock(object):
    """
    parameters:
    -----------
    conn: connection to database
        e.g. import sqlite3 as sql
             conn = sql.connect('datapath/data.db')    
    code; stock code
    start: start date
    end: end date
    """
    def __init__(self, conn, code, start='2017-01-01', end='2017-12-01'):
        self._code = code
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._conn = conn
        self._price = self.get_data(conn)
    
    def get_data(self, conn):
        c = conn.cursor()
        c.execute('select * from stocks_price where code="%s"' % (self._code))
        stocks_price = pd.DataFrame(c.fetchall())
        if len(stocks_price)==0:
            raise ValueError('no data fetched')
        stocks_price.columns = ['index', 'date', 'open', 'close', 'high', 'low', 'volume', 'code']
        stocks_price['date'] = pd.to_datetime(stocks_price['date'])
        stocks_price.set_index('date', inplace=True)
        if self.check_date(stocks_price):
            self._price = stocks_price.iloc[(stocks_price.index >= self._start) 
                                            & (stocks_price.index <= self._end), :]
        else:
            raise ValueError('data start from {} and end at {}'.format(('data_start', 'data_end')))
        
        return self._price
    
    def check_date(self, data):
        data_start = data.index.min()
        data_end = data.index.max()
        if (data_start>self._start) | (data_end<self._end):
            return False
        else:
            return True
    
    @property
    def code(self):
        return self._code
    
    @property
    def price(self):
        return self._price
    
    @property    
    def close_price(self):
        df = pd.DataFrame.assign(self._price)
        self._close_price = df[['close']]
        return self._close_price
        
    @property
    def daily_returns(self):
        df = pd.DataFrame.assign(self._price)
        df['return'] = np.log(df['close'] / df.groupby(by='code')['close'].shift(1))
        df.dropna(axis=0, how='any', inplace=True)
        self._daily_returns = df[['return']]
        self._daily_returns.columns = ['daily_return']
        return self._daily_returns
    
    @property
    def daily_cum_returns(self):
        self._daily_cum_returns = np.log(self._close_price/self._close_price.iloc[0])
        self._daily_cum_returns.columns = ['daily_cum_returns']
        return self._daily_cum_returns
    
