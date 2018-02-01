
# coding: utf-8

# In[1]:


#import numpy as np
#import pandas as pd
#import pandas as pandas
#import sqlite3
#import matplotlib.pyplot as plt
# import sys
# sys.path.append('/Users/zifandeng/Nustore Files/PI/Code/Working_On/')
#from Working_On.Ratios import *
#from Working_On.stock_class import Stock


# In[119]:


import numpy as np
import pandas as pd
from stock_class import Stock

#%%


class Portfolio(object):
    """
    parameters:
    -----------
    stocks_list: a dataframe contains stock codes and shares
            e.g. stocks_list = pd.DataFrame({'code': ['000001','000002'],
                                             'shares': [1000, 1000]})
    """

    def __init__(self, conn, code_list, start, end):
        self._conn = conn
        self._code_list = code_list
        self._start = start
        self._end = end
        self._price, self._returns = self.construct()
        self._benchmark = self.add_benchmark()

    def construct(self):
        price = pd.DataFrame()
        returns = pd.DataFrame()
        for code in self._code_list['code']:
            stock_data = Stock(self._conn, code, self._start, self._end)
            stock_price = stock_data.close_price
            stock_price.columns = [code]
            stock_return = stock_data.daily_returns
            stock_return.columns = [code]
            price = self.add_stock(price, stock_price)
            returns = self.add_stock(returns, stock_return)
        return price, returns

    def add_stock(self, port, stock):
        if len(port) == 0:
            port = stock
        else:
            port = port.merge(stock, left_index=True, right_index=True, how='outer')
        return port

    def port_price(self):
        return self._price

    def stock_returns(self):
        return self._returns

    def port_returns(self):
        port_return = pd.DataFrame({'Portfolio': np.log(self.port_daily_balance()) -
                                    np.log(self.port_daily_balance().shift(1))})
        port_return.dropna(axis=0, how='any', inplace=True)
        return port_return

    def add_benchmark(self, index_code='sh000001', price_type='close'):
        c = self._conn.cursor()
        c.execute('select * from index_price where code="%s"' % (index_code))
        index = pd.DataFrame(c.fetchall())
        if len(index) == 0:
            raise ValueError('no data fetched')
        index.columns = ['index', 'date', 'open', 'close', 'high', 'low', 'volume', 'code', 'name']
        index['date'] = pd.to_datetime(index['date'])
        index.set_index('date', inplace=True)
        index = index.iloc[(index.index >= self._start)
                           & (index.index <= self._end), :]
        index = index[[price_type, 'code', 'name']]
        index.columns = ['price', 'code', 'name']
        self._benchmark = index
        return index

    def benchmark(self):
        benchmark = self._benchmark
        return benchmark

    def benchmark_info(self):
        return (self.benchmark()[['code', 'name']]).head(1)

    def benchmark_returns(self):
        price = self.benchmark()['price']
        index_return = pd.DataFrame({'index': np.log(price) - np.log(price.shift(1))})
        index_return.dropna(axis=0, how='any', inplace=True)
        return index_return

    def port_initial_shares(self):
        Shares = pd.DataFrame.assign(self._code_list)
        Shares.set_index('code', inplace=True)
        return Shares

    def stock_daily_balance(self):
        shares = np.array(self._code_list['shares'])
        balance = self.port_price().rmul(shares)
        balance = balance.fillna(method='ffill')
        return balance

    def port_daily_balance(self):
        stock_balance = self.stock_daily_balance()
        balance = (stock_balance).sum(axis=1)
        return balance

    def port_allocation(self):
        initial_price = pd.DataFrame({'initial_price': self.port_price().iloc[0, :]})
        shares = self.port_initial_shares()
        df = initial_price.merge(shares, left_index=True, right_index=True, how='outer')
        df['market_cap'] = df['initial_price'] * df['shares']
        df['total'] = np.sum(df['market_cap'])
        df['allocation'] = df['market_cap'] / df['total']
        return df[['initial_price', 'shares', 'market_cap', 'allocation']]

    def allocation_plot(self):
        df = self.port_allocation()
        get_ipython().magic('matplotlib inline')
        ax = plt.figure()
        plt.axes(aspect='equal')
        plt.pie(df['allocation'], autopct='%.1f%%')
        plt.legend(df.index, loc='lower right')
        plt.title('Portfolio Allocation')
        return

    def port_balance_plot(self):
        get_ipython().magic('matplotlib inline')
        plt.figure()
        balance = self.port_daily_balance()
        balance.plot(kind='area', grid=True, title='Portfolio Balance')

    def performance_matrix(self, annualization=252):
        ret_price = pd.merge(self.stock_returns(), self.port_returns(), left_index=True, right_index=True)
        ret_index = self.benchmark_returns()['index']
        matrix = pandas.DataFrame({'Beta': ret_price.apply(get_beta, market=ret_index),
                                   'Annualized Alpha': ret_price.apply(get_annulized_alpha, market=ret_index, annualization=annualization),
                                   'R Square': ret_price.apply(get_rsquare, market=ret_index),
                                   'Adj R Square': ret_price.apply(get_adj_rsquare, market=ret_index),
                                   'Martket Correlation': ret_price.apply(get_correlation, market=ret_index),
                                   'Sharpe Ratio': ret_price.apply(get_sharpe_ratio, annualization=annualization),
                                   'Sortino Ratio': ret_price.apply(get_sortino_ratio, annualization=annualization),
                                   'Treynor Ratio': ret_price.apply(get_treynor_ratio, market=ret_index, annualization=annualization),
                                   'Positive Period': ret_price.apply(positive_period),
                                   'Gain/Loss Ratio': ret_price.apply(gain_loss_ratio)})
        return matrix

    def port_summary(self):
        summary = self.port_allocation()
        summary['allocation'] = round(summary['allocation'], 4) * 100
        summary.columns = ['Initial Price', 'Shares', 'Initial Balance', 'Allocation(%)']
        summary['End Balance'] = self.stock_daily_balance().loc[self._end, :]
        summary.loc['Grand Total', :] = summary.sum(0)
        return summary

    def port_performance_matrix(self):
        # Include ratios from performance matrix
        ratios = pd.DataFrame(self.performance_matrix().loc['Portfolio', :])

        # Calculater Other Factors like: VaR, Volatility, Annual Return, etc
        index_all = ['Diversification Ratio']
        # 1. Diversification Ratio
        weight = self.port_allocation()['allocation']
        div_ratio = get_diversification_ratio(stock_ret=self.stock_returns(), portfolio_ret=self.port_returns(), weight=weight)
        # 2. VaRs
        # 3. Returns
        # 4. Vols
        # Return factor
        perfomance_factors = pd.DataFrame([div_ratio], index=index_all)
        perfomance_factors.columns = ['Portfolio']
        # bind
        port_matrix = pd.concat([ratios, perfomance_factors], axis=0)

        return port_matrix
