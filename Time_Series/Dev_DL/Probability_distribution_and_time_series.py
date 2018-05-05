
# coding: utf-8

# In[22]:


import pandas as pd
import sqlite3 as sql
import numpy as np
import scipy as sp
import matplotlib.pylab as plt


# In[5]:


# smimulate i.i.d random variable of t with location-scale parameters
sim_data = pd.Series
sim_data.mu = 0.01
sim_data.sigma = 0.03
sim_data.df = 4
sim_data.num = 10000
sim_data = sim_data.mu+ np.random.standard_t(sim_data.df,sim_data.num)*sim_data.sigma
std_data = (sim_data-np.mean(sim_data))/np.std(sim_data)


# In[ ]:





# In[91]:


# fit emperiocal data to best distribution

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [  
        st.norm,st.t
    ]


    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    
    # Standardize data
    std_data = (data-np.mean(data))/np.std(data)

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(std_data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)



# In[129]:


# Simulation on best distribution
fitted_dist = best_fit_distribution(sim_data)
best_dist = getattr(st, fitted_dist[0])
best_dist.params = fitted_dist[1]
best_dist.loc = fitted_dist[1][-2]
best_dist.scale = fitted_dist[1][-1]
fitted_dist = best_dist.rvs(best_dist.params[:-2],best_dist.loc,best_dist.scale,1000)


# In[164]:


# KS test
D, p_value = st.stats.kstest(std_data,'t',best_dist.params,alternative='two-sided')


# In[ ]:


# fit ARMA-GARCH model


# In[1]:


from arch import arch_model


# In[9]:


# create a DateFrame with index
starting_date = '2010-1-1'
dates = pd.date_range(starting_date, periods = std_data.size, freq='D')
ts_data = pd.DataFrame(std_data,columns = ['iid t'], index = dates)


# In[20]:


# check stationarity 
# constant mean
# constant variance
# an autocovariance that does not depend on time.
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.DataFrame.rolling(timeseries,window=12,center=False).mean()
    rolstd = pd.DataFrame.rolling(timeseries,window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[25]:


test_stationarity(ts_data['iid t'])


# In[29]:


# make timeseries stationary
# 1. Trend – varying mean over time. For eg, in this case we saw that on average, the number of passengers was growing over time.
# 2. Seasonality – variations at specific time-frames. eg people might have a tendency to buy cars in a particular month because of pay increment or festivals.

# Moving average to calculate trend
ts_moving_avg = pd.DataFrame.rolling(ts_data,window=12,center=False).mean()
ts_moving_avg_diff = ts_data - ts_moving_avg
ts_moving_avg_diff.dropna(inplace=True)

# exponentially weighted moving average
ts_expwighted_avg = pd.DataFrame.ewm(ts_data,halflife=12,min_periods=0,adjust=True,ignore_na=False).mean()


# In[30]:


# Eliminating Trend and Seasonality
# Differencing – taking the differece with a particular time lag

ts_data_diff = ts_data - ts_data.shift()
ts_data_diff.dropna(inplace=True)



# In[32]:


# Decomposition – modeling both trend and seasonality and removing them from the model.
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_data)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

ts_data_decompose = residual
ts_data_decompose.dropna(inplace=True)


# In[50]:


#  Forecasting a Time Series

# ACF and PACF
#ACF and PACF plots:


from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_data, nlags=20)
lag_pacf = pacf(ts_data, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[48]:


# AR p = 2
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_data, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  

# MA q = 2
model = ARIMA(ts_data, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  

# ARMA p,q = 2,2
model = ARIMA(ts_data, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  

# Prediction
predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)

# In[327]:


# select ARIMA estimation method and solver
# ARMA p,q = 1,1
METHOD = ['css-mle','mle','css']
SOLVER = ['lbfgs','newton','nm','cg','ncg','powell']

#%%
model = ARMA(ts_data,order=(2,2)) 
best_aic = np.inf
n = 0
aic_list = np.zeros((1,18))
res_sq_list =np.zeros((1,18))
for mtd in METHOD:
    
    for sol in SOLVER:
        
        results = ARMA.fit(model,start_params = None,trend = 'c', method = mtd, solver = sol, maxiter = 100,disp = 1, order=(2,2))  
        aic = results.aic
        res = results.resid
         # identify if this fitting is better
        if aic < best_aic :
            
            best_method = mtd
            best_solver = sol
            best_results = results
            aic_list[0,n]= aic
            res_sq_list[0,n] = sum(i*i for i in res)
            
        n = n+1
        
    
            

# In[51]:


# predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
ts_log = np.log(ts)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))


# In[53]:


# GARCH (with a Constant Mean)from arch import arch_model
am = arch_model(ts_data,p = 1,q = 1) 
res = am.fit(update_freq=5)
print(res.summary())
res.plot(annualize='D')


# In[56]:


res.plot(annualize='D')


# In[58]:


# AR
from arch.univariate import ARX
ar = ARX(ts_data, lags = [1, 3, 12])
# print(ar.fit().summary())


# In[60]:


# Volatility Processes
from arch.univariate import ARCH, GARCH
ar.volatility = ARCH(p=5)
res = ar.fit(update_freq=0, disp='off')
# print(res.summary())
fig = res.plot()

# Distribution
from arch.univariate import StudentsT
ar.distribution = StudentsT()
res = ar.fit(update_freq=0, disp='off')
# print(res.summary())


# In[61]:


# price to return
# crude_ret = 100 * crude.dropna().pct_change().dropna()


# In[62]:


from collections import OrderedDict

res_normal = arch_model(ts_data).fit(disp='off')
res_t = arch_model(ts_data, dist='t').fit(disp='off')
res_skewt = arch_model(ts_data, dist='skewt').fit(disp='off')
lls = pd.Series(OrderedDict((('normal', res_normal.loglikelihood),
                 ('t', res_t.loglikelihood),
                 ('skewt', res_skewt.loglikelihood))))
print(lls)
params = pd.DataFrame(OrderedDict((('normal', res_normal.params),
                 ('t', res_t.params),
                 ('skewt', res_skewt.params))))
print(params)


# In[63]:


# The standardized residuals can be computed by dividing the residuals by the conditional volatility. These are plotted along with the (unstandardized, but scaled) residuals. The non-standardized residuals are more peaked in the center indicating that the distribution is somewhat more heavy tailed than that of the standardized residuals.
std_resid = res_normal.resid / res_normal.conditional_volatility
unit_var_resid = res_normal.resid / res_normal.resid.std()
df = pd.concat([std_resid, unit_var_resid],1)
df.columns = ['Std Resids', 'Unit Variance Resids']
df.plot(kind='kde', xlim=(-4,4))


# In[65]:




