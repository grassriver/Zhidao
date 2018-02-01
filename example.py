#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:14:43 2018

@author: tianyulu
"""

from simplified_var_functions import *
from Ratios import *
from max_drawdown import *
import pandas as pd

' Create connextion
conn = sql.connect('/Users/tianyulu/Nustore Files/PI/Staff Working File/D.Lu/data.db')

' select a stock
ticker = '000001'
a_stock = Stock(conn,ticker)

' stock return
a_stock_ret = a_stock.daily_returns()

' distribution statistics
print(a_stock_ret.skew())
print(a_stock_ret.kurtosis())

a_plot = a_stock_ret.plot(kind = 'density',title = ticker)
a_plot.set_xlabel("daily return")


' calculate var

hist_var(a_stock_ret,cutoff = 0.05)
ewma_var(a_stock_ret,cutoff = 0.05,span_value = 1.0001)

' calculate drawdown
gen_drawdown_table(a_stock_ret.daily_return)
' plot drawdown
plot_drawdown_underwater(a_stock_ret.daily_return)

' Portfolio



