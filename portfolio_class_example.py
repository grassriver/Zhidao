
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pandas as pandas
import sqlite3
import matplotlib.pyplot as plt
# import sys
# sys.path.append('/Users/zifandeng/Nustore Files/PI/Code/Working_On/')
from Working_On.Ratios import *
from Working_On.stock_class import Stock
from Working_On.portfolio_class_modified import Portfolio
import numpy as np
import pandas as pd


# In[2]:


# conn = sqlite3.connect('/Users/zifandeng/Nustore Files/PI/data/data.db')
conn = sqlite3.connect('d:/Kaizheng/Working_directory/portfolio_intelligence/PI/data/data.db')
stocks_list = pd.DataFrame({'code': ['000001', '000002', '600015'], 'shares': [5000, 1000, 1000]})
port = Portfolio(conn, stocks_list, '2017-01-01', '2017-12-01')


# In[3]:


port.port_summary()


# In[4]:


port.port_balance_plot()


# In[5]:


port.allocation_plot()


# In[6]:


# See Benchmark Info
port.benchmark_info()


# In[7]:


port.performance_matrix()


# In[8]:


port.port_performance_matrix()


# In[9]:


# Change Benchmark
port.add_benchmark('sh000002')
# See Benchmark Info
port.benchmark_info()


# In[10]:


# matrix will change accordingly
port.performance_matrix()


# In[11]:


port.port_performance_matrix()
