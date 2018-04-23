import numpy as np
import pandas as pd
from stock_class import Stock
import Ratios as ratio
import VaR as var
import weekly_monthly_returns as wmr
import matplotlib.pyplot as plt
import max_drawdown as md
import candlestick as cstick
import warnings

#%%
class Portfolio(object):
    """
    parameters:
    -----------
    stocks_list: a dataframe contains stock codes and shares
            e.g. stocks_list = pd.DataFrame({'code': ['000001','000002'],
                                             'shares': [1000, 1000]})
    """

    def __init__(self, conn, code_list, start='2017-01-01', end='2017-12-01',
                 annualization=252, backfill=False, stocks_price_old=None,
                 business_calendar=None, industry=None, benchmark_old=None):
        self._conn = conn
        self._code_list = code_list
        self._start = start
        self._end = end
        self._backfill = backfill
        self._stocks_price_old = stocks_price_old
        self._business_calendar = business_calendar
        self._industry = industry
        self._benchmark_old = benchmark_old
        self._price, self._returns, self._industry = self.construct()
        self._benchmark = self.add_benchmark()
        self._annualization = annualization

    def construct(self):
        price = pd.DataFrame()
        returns = pd.DataFrame()
        industry = pd.DataFrame()
#        for code in self._code_list['code']:
        code_list = list(self._code_list['code'])
        stock_data = Stock(self._conn, code_list, self._start, self._end,
                           backfill=self._backfill, 
                           stocks_price_old=self._stocks_price_old,
                           business_calendar=self._business_calendar,
                           industry = self._industry)
        stock_price = stock_data.close_price
#        stock_price.columns = [code]
        stock_return = stock_data.daily_returns
        stock_industry = stock_data.industry
#        stock_return.columns = [code]
        price = self.add_stock(price, stock_price)
        returns = self.add_stock(returns, stock_return)
        industry = self.add_stock(industry, stock_industry)
        return price, returns, industry

    def add_stock(self, port, stock):
        if len(port) == 0:
            port = stock
        else:
            port = port.merge(stock, left_index=True, right_index=True, how='outer')
        return port

    def industry(self):
        return self._industry

    def stock_price(self):
        return self._price

    def stock_returns(self):
        return self._returns

    def weekly_returns(self):
        port = pd.DataFrame({'portfolio': self.port_daily_balance()})
        price = pd.merge(self.stock_price(), port, left_index=True, right_index=True)
        return wmr.weekly_return(price)

    def monthly_returns(self):
        port = pd.DataFrame({'portfolio': self.port_daily_balance()})
        price = pd.merge(self.stock_price(), port, left_index=True, right_index=True)
        return wmr.monthly_return(price)

    def port_returns(self):
        port_return = pd.DataFrame({'Portfolio': np.log(self.port_daily_balance()) -
                                    np.log(self.port_daily_balance().shift(1))})
        port_return.dropna(axis=0, how='any', inplace=True)
        return port_return

    def add_benchmark(self, index_code='sh000001', price_type='close'):
        if self._benchmark_old is None:
            query = 'select * from index_price where code="%s"' % (index_code)
            index = pd.read_sql(query, self._conn)
        else:
            index = self._benchmark_old.copy()
        if len(index) == 0:
            raise ValueError('no data fetched')
        index['date'] = pd.to_datetime(index['date'])
        index.set_index('date', inplace=True)
        index = index.iloc[(index.index >= self._start)
                           & (index.index <= self._end), :]
        index = index.sort_index()
        index = index[[price_type, 'code', 'name']]
        index.columns = ['price', 'code', 'name']
        return index
            

    def change_benchmark(self, index_code, price_type='close'):
        query = 'select * from index_price where code="%s"' % (index_code)
        index = pd.read_sql(query, self._conn)

        if len(index) == 0:
            raise ValueError('no data fetched')
        index['date'] = pd.to_datetime(index['date'])
        index.set_index('date', inplace=True)
        index = index.iloc[(index.index >= self._start)
                           & (index.index <= self._end), :]
        index = index.sort_index()
        index = index[[price_type, 'code', 'name']]
        index.columns = ['price', 'code', 'name']
        self._benchmark = index
        
    def benchmark(self):
        benchmark = self._benchmark
        return benchmark

    @property
    def benchmark_info(self):
        return (self.benchmark()[['code', 'name']]).head(1)

    def benchmark_returns(self):
        price = self.benchmark()['price']
        index_return = pd.DataFrame({'index': np.log(price) - np.log(price.shift(1))})
        index_return.dropna(axis=0, how='any', inplace=True)
        index_return = pd.merge(index_return, self.port_returns(), left_index=True, right_index=True, how='right')
        return index_return[['index']]

    def port_initial_shares(self):
        Shares = pd.DataFrame.assign(self._code_list)
        Shares.set_index('code', inplace=True)
        return Shares

    def stock_daily_balance(self):
        shares = np.array(self._code_list['shares'])
        stock_price = self.stock_price()[self._code_list['code']]
        balance = stock_price.rmul(shares)
        balance = balance.fillna(method='ffill')
        return balance

    def port_daily_balance(self):
        stock_balance = self.stock_daily_balance()
        balance = (stock_balance).sum(axis=1)
        return balance

    def nav_plot(self):
        port_balance = pd.DataFrame(self.port_daily_balance())
        port_nav = port_balance / port_balance.iloc[0, 0]
        port_nav.columns = ['portfolio']
        bench_balance = self.benchmark()[['price']]
        bench_nav = bench_balance / bench_balance.iloc[0, 0]
        bench_nav.columns = ['benchmark']
        nav = port_nav.merge(bench_nav, how='left', left_index=True, right_index=True)
        fig = nav.plot()
        plt.title('portfolio return vs benchmark return')
        return fig

    def port_allocation(self):
        initial_price = pd.DataFrame({'initial_price': self.stock_price().iloc[0, :]})
        shares = self.port_initial_shares()
        df = initial_price.merge(shares, left_index=True, right_index=True, how='outer')
        df['market_cap'] = df['initial_price'] * df['shares']
        df['total'] = np.sum(df['market_cap'])
        df['allocation'] = df['market_cap'] / df['total']
        df = df.sort_index()
        return df[['initial_price', 'shares', 'market_cap', 'allocation']]

    def industry_allocation(self):
        allocation = round(self.port_allocation() * 100, 2)
        industry = self.industry()
        df = pd.merge(industry, allocation, left_index=True, right_index=True, how='left')
        df = df[['industry', 'allocation']]
        df.columns = ['Industry', 'Allocation%']
        df = df.groupby('Industry').sum()
        df['Sector_Code'] = range(1, len(df) + 1)
        #self.allocation_plot(types = 'industry')
        return df[['Sector_Code', 'Allocation%']]

    def allocation_plot(self, types='stocks'):
        plt.figure()
        plt.axes(aspect='equal')
        if(types == 'stocks'):
            df = self.port_allocation()
            plt.pie(df['allocation'], autopct='%.1f%%')
            plt.legend(df.index, loc='lower right')
            plt.title('Portfolio Allocation')
        else:
            plt.pie(self.industry_allocation()['Allocation%'], autopct='%.1f%%')
            plt.legend(self.industry_allocation()['Sector_Code'], loc='lower right')
            plt.title('Industry Allocation')
        plt.show()

    def port_balance_plot(self):
        plt.figure()
        balance = self.port_daily_balance()
        balance.plot(kind='area', grid=True, title='Portfolio Balance')

    def performance_matrix(self):
        annualization = self._annualization
        ret_price = pd.merge(self.stock_returns(), self.port_returns(), left_index=True, right_index=True)
        ret_index = self.benchmark_returns()['index']
        matrix = pd.DataFrame({'Beta': ret_price.apply(ratio.get_beta, market=ret_index),
                               'Annualized Alpha': ret_price.apply(ratio.get_annulized_alpha, market=ret_index, annualization=annualization),
                               'R Square': ret_price.apply(ratio.get_rsquare, market=ret_index),
                               'Adj R Square': ret_price.apply(ratio.get_adj_rsquare, market=ret_index),
                               'Market Correlation': ret_price.apply(ratio.get_correlation, market=ret_index),
                               'Sharpe Ratio': ret_price.apply(ratio.get_sharpe_ratio, annualization=annualization),
                               'Sortino Ratio': ret_price.apply(ratio.get_sortino_ratio, annualization=annualization),
                               'Treynor Ratio': ret_price.apply(ratio.get_treynor_ratio, market=ret_index, annualization=annualization),
                               'Positive Period': ret_price.apply(ratio.positive_period),
                               'Gain/Loss Ratio': ret_price.apply(ratio.gain_loss_ratio),
                               'Historical VaR': ret_price.apply(var.hist_var),
                               'Delta-Normal VaR': ret_price.apply(var.delta_normal_var),
                               'EWMA VaR': ret_price.apply(var.ewma_var),
                               'Conditional VaR': ret_price.apply(var.conditional_var),
                               'Kurtosis': round(ret_price.kurtosis(), 4),
                               'Skewness': round(ret_price.skew(), 4),
                               'Daily Volatility': ret_price.apply(ratio.volatility)})
        return matrix

    def port_summary(self):
        summary = self.port_allocation()

        summary['allocation'] = round(summary['allocation'], 4) * 100
        summary.columns = ['Initial Price', 'Shares', 'Initial Balance', 'Allocation(%)']
        summary['End Balance'] = self.stock_daily_balance().tail(1).iloc[0, ]
        summary.loc['Grand Total', :] = summary.sum(0)
        return summary

    def port_performance_matrix(self):
        # Include ratios from performance matrix
        ratios = pd.DataFrame(self.performance_matrix().loc['Portfolio', :])

        # Calculater Other Factors like: VaR, Volatility, Annual Return, etc
        index_all = ['Diversification Ratio', 'Max Drawdown']
        # 1. Diversification Ratio
        weight = self.port_allocation()['allocation']
        div_ratio = ratio.get_diversification_ratio(stock_ret=self.stock_returns(), portfolio_ret=self.port_returns(), weight=weight)
        # 2. VaRs
        drawdown = round(self.gen_drawdown_table().iloc[0, 0], 4)
        # 3. Returns
        # 4. Vols
        # Return factor
        perfomance_factors = pd.DataFrame([div_ratio[0], drawdown], index=index_all)
        perfomance_factors.columns = ['Portfolio']
        # bind
        port_matrix = pd.concat([ratios, perfomance_factors], axis=0)

        return port_matrix

    def gen_drawdown_table(self, top=5):
        drawdown_table = md.gen_drawdown_table(self.port_returns().iloc[:, 0], top)
        return drawdown_table

    def plot_drawdown_periods(self, top=5, ax=None, **kwargs):
        ax = md.plot_drawdown_periods(self.port_returns().iloc[:, 0], top, ax, **kwargs)
        return ax

    def plot_drawdown_underwater(self, ax=None, **kwargs):
        ax = md.plot_drawdown_underwater(self.port_returns().iloc[:, 0], ax, **kwargs)
        return ax

    def candle_stick_plot(self, code, stick='day', see='n'):
        [fig, no_traing_days] = cstick.stock_plot(self._conn, stick, code, self._start, self._end, see)
        return fig, no_traing_days
