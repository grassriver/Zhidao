# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:54:35 2018

@author: Think
"""
import numpy as np
import pandas as pd
import max_drawdown as md

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

    def get_data(self, conn):
        c = conn.cursor()
        if isinstance(self._code, (type, str)):
            c.execute('select * from stocks_price where code="%s"' % (self._code))
        elif isinstance(self._code, (type, list)):
            c.execute('select * from stocks_price where code in ("%s")' % ('","'.join(self._code)))
        else:
            raise ValueError('code can only be a string or a list')
             
        stocks_price = pd.DataFrame(c.fetchall())
        if len(stocks_price) == 0:
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
        if (data_start > self._start) | (data_end < self._end):
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
        """
        return a pandas series of close price
        """
        df = pd.DataFrame.assign(self._price)
        self._close_price = df.pivot(columns='code', values='close')
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
        self._daily_cum_returns = np.log(self._close_price / self._close_price.iloc[0])
        self._daily_cum_returns.columns = ['daily cumulative returns']
        return self._daily_cum_returns
    
    def gen_drawdown_table(self, top=10):
        drawdown_table = md.gen_drawdown_table(self.daily_returns['daily returns'], top)
        return drawdown_table
    
    def plot_drawdown_periods(self, top=10, ax=None, **kwargs):
        ax = md.plot_drawdown_periods(self.daily_returns['daily returns'], top, ax, **kwargs)
        return ax
    
    def plot_drawdown_underwater(self, ax=None, **kwargs):
        ax = md.plot_drawdown_underwater(self.daily_returns['daily returns'], ax, **kwargs)
        return ax
    
        
