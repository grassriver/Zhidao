
# coding: utf-8

# In[1]:


import sqlite3 as sql
import pandas as pd
import sys
sys.path.append('/Users/zifandeng/Nustore Files/PI/Code/Working_On')
from stock_class import Stock
from portfolio_class import Portfolio
import stock_screener as sc
import matplotlib.pyplot as plt
import general_tools as tool


# In[2]:


conn_path='/Users/zifandeng/Documents/'
date = '2015-10-31'


# In[3]:


stock_list = sc.stock_screener_filter_top(conn_path = conn_path,date=date,var_list=['净利率 (%)'],top=[70,20,100],
                             order=['ascending','ascending','ascending'],industry = ['机械','传媒','金融'],in_universe = False,since_ipo = {'min':0.75,'max':1})
sc.print_data(conn_path,stock_list)


# In[17]:


stock_list = sc.stock_screener_filter_condition(conn_path,date=date,var_list=['净利率 (%)','净资产收益 (%)','每股收益'],
                                                threshold_list= [0,0,0],
                                                condition_list=['>=','>=','>='],
                                                since_ipo = {'min':0.0,'max':5.75},
                                                industry = ['传媒','金融','建筑','房地产'])
sc.print_data(conn_path,stock_list)


# In[18]:


stock_list=sc.stock_screener_ranking(conn_path=conn_path,date=date,var_list=['净利率 (%)','净资产收益 (%)','每股收益'],rank_by = '每股收益',order = 'decending',top=10,in_universe=False,industry=['机械'],since_ipo = {'min':0.75,'max':1})
sc.print_data(conn_path,stock_list)


# In[4]:


conn=sql.connect('/Users/zifandeng/Nustore Files/PI/data/data.db')


# In[6]:


stocks = tool.portfolio_construct(conn,start = '2016-04-11',code_list=stock_list.Code,
                                  name_list=stock_list.Name,construct_type='weight',equal=True)


# In[8]:


P = Portfolio(conn,stocks,start='2016-04-11',end = '2016-06-30')


# In[9]:


P.allocation_plot()
plt.show()


# In[29]:


P.port_summary()


# In[51]:


P.port_balance_plot()
plt.show()


# In[52]:


P.nav_plot()
plt.show()


# In[ ]:


P.plot_drawdown_periods()
plt.show()


# In[ ]:


P.plot_drawdown_underwater()
plt.show()


# In[ ]:


P.gen_drawdown_table()


# In[ ]:


P.weekly_returns()


# In[ ]:


P.monthly_returns()


# In[ ]:


P.performance_matrix()


# In[19]:


P.port_performance_matrix()

