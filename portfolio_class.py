# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:52:52 2018

@author: Think
"""

import numpy as np
import pandas as pd
from .stock_class import Stock

#%%
class Portfolio(object):
    """
    parameters:
    -----------
    stocks_list: a dataframe contains stock codes and shares
            e.g. code_list = pd.DataFrame({'code': ['000001','000002'],
                                             'shares': [1000, 1000]})
    """
    def __init__(self, conn, code_list):
        self._conn = conn
        self._code_list = code_list
        self._price, self._returns = self.construct()
        
    def construct(self):
        price = pd.DataFrame()
        returns = pd.DataFrame()
        for code in self._code_list['code']:
            stock_data = Stock(self._conn,code)
            stock_price = stock_data.close_price
            stock_price.columns = [code]
            stock_return = stock_data.daily_returns
            stock_return.columns = [code]
            price = self.add_stock(price, stock_price)
            returns = self.add_stock(returns, stock_return)

        return price, returns
        
    def add_stock(self, port, stock):
        if len(port)==0:
            port = stock
        else:
            port = port.merge(stock, left_index=True, right_index=True, how='outer')
        return port
    
    @property
    def price(self):
        return self._price
    
    @property
    def daily_returns(self):
        return self._returns
    
    def balance(self):
        pass