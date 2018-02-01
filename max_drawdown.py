# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 09:56:27 2017

@author: Kai
"""

import numpy as np
import pandas as pd
import sqlite3 as sql
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore',
        'axes.color_cycle is deprecated',
        UserWarning,
        'matplotlib',
    )
    import seaborn as sns  # noqa
    
APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    YEARLY: 1
}
    
#%%
# =============================================================================
# generate daily returns
# =============================================================================
def gen_return(df):
    """
    generate return from the price
    """
    df['return'] = np.log(df['close']/df.groupby(by='code')['close'].shift(1))
    df.dropna(axis=0, how='any', inplace=True)
    return df

# =============================================================================
# calculate annual return
# =============================================================================

def annualization_factor(period, annualization):
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor

def cum_returns(returns, starting_value):
    if len(returns) < 1:
        return type(returns)([])

    if np.any(np.isnan(returns)):
        returns = returns.copy()
        returns[np.isnan(returns)] = 0.

    df_cum = (returns + 1).cumprod(axis=0)

    if starting_value == 0:
        return df_cum - 1
    else:
        return df_cum * starting_value

def annual_return(returns, period=DAILY, annualization=None):
    if len(returns)<1:
        return np.nan
    
    ann_factor = annualization_factor(period, annualization)
    num_years = float(len(returns))/ann_factor
    start_value = 100
    end_value = cum_returns(returns, starting_value=start_value)[-1]
    cum_returns_final = (end_value - start_value) / start_value
    annual_return = (1. + cum_returns_final) ** (1. / num_years) - 1

    return annual_return

# =============================================================================
# ep.cum_returns_final
# =============================================================================
def cum_return_final(returns, starting_value=0):
    if len(returns) == 0:
        return np.nan

    return cum_returns(np.asanyarray(returns),
                       starting_value=starting_value)[-1]
    
# =============================================================================
# max_drawdown
# =============================================================================
def get_max_drawdown_underwater(underwater):
    """
    Determines peak, valley, and recovery dates given an 'underwater'
    DataFrame.

    An underwater DataFrame is a DataFrame that has precomputed
    rolling drawdown.

    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.

    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """

    valley = underwater.idxmin()  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery

def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum) # cumulative max of cum_return
    underwater = df_cum / running_max - 1

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        if underwater.loc[valley]==0:
            break
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0):
            break

    return drawdowns


def gen_drawdown_table(returns, top=10):
    """
    Places top drawdowns in a table.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """

    df_cum = cum_returns(returns, 1.0) #cumulative returns
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(len(drawdown_periods))),
                                columns=['Net drawdown in %',
                                         'Peak date',
                                         'Valley date',
                                         'Recovery date',
                                         'Duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'Duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                recovery,
                                                                freq='B'))
        df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
                                            .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
                                              .strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'Recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Net drawdown in %'] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100

    df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
    df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
    df_drawdowns['Recovery date'] = pd.to_datetime(
        df_drawdowns['Recovery date'])

    return df_drawdowns
# =============================================================================
# plot drawdowns
# =============================================================================
def two_dec_places(x, pos):
    """
    Adds 1/100th decimal to plot ticks.
    """

    return '%.2f' % x

def plot_drawdown_periods(returns, top=10, ax=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = cum_returns(returns, starting_value=1.0)
    df_drawdowns = gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['Peak date', 'Recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])
    ax.set_ylim(lim)
    ax.set_title('Top %i drawdown periods' % top)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Portfolio'], loc='upper left',
              frameon=True, framealpha=0.5)
    ax.set_xlabel('')
    return ax

def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return '%.0f%%' % x

def plot_drawdown_underwater(returns, ax=None, **kwargs):
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)
    ax.set_ylabel('Drawdown')
    ax.set_title('Underwater plot')
    ax.set_xlabel('')
    return ax

#%%
#conn = sql.connect('D:/Kaizheng/Working_directory/portfolio_intelligence/data/tushare/data.db')
#c = conn.cursor()
#
#c.execute('select * from stocks_price')
#stocks_price = pd.DataFrame(c.fetchall())
#stocks_price.columns = ['index', 'date', 'open', 'close', 'high', 'low', 'volume', 'code']
#
#stocks_return = gen_return(stocks_price)
#stocks_return.set_index(pd.DatetimeIndex(stocks_return.date),inplace=True)
#
#my_return = stocks_return.loc[stocks_return.code.isin(['000002']), 'return']
#print(annual_return(my_return))
#
#df_drawdowns = gen_drawdown_table(my_return)
#print(df_drawdowns)
#
#plot_drawdown_periods(my_return)
#plot_drawdown_underwater(my_return)
