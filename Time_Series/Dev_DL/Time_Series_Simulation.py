
# coding: utf-8

"""
Created on Sat Mar 17 11:21:36 2018

@author: tianyulu
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import sys

# Simulate a ARMA-GARCH model with Normal and Student t innovations

np.random.seed(1)
n_samples = 1000

z = np.random.normal(size=n_samples)
x = np.ones((n_samples,))
s = np.ones((n_samples,))

m,kappa, theta = [0.05,-0.3,0.2]



for t in range(n_samples):
    s[t] = np.sqrt(0.01+0.95*s[t-1]*s[t-1]+0.02*s[t-1]*w[t-1])
    x[t] = 0.8*x[t-1] - 0.05*w[t-1]*s[t-1]+w[t]*s[t]

ts_data = x
plt.plot(ts_data)
plt.show()



np.shape(results_ARIMA.fittedvalues)



# Fit ARMA model
# ACF and PACF
#ACF and PACF plots:


from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

lag_acf = acf(ts_data, nlags=20)
lag_pacf = pacf(ts_data, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()




# Fit ARMA (1,1)
# ARMA p,q = 1,1
model = ARIMA(ts_data, order=(1, 0, 1))  
results_ARIMA = model.fit(disp=-1)  

# Prediction
predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)

# Residule
residual_ARIMA = ts_data-results_ARIMA.fittedvalues


print(results_ARIMA.summary())



plt.plot(results_ARIMA.fittedvalues, color='blue')
plt.plot(ts_data[1:], color='red')

plt.show()




# Fit GARCH model 
# GARCH (with a Constant Mean)from arch import arch_model

from arch import arch_model

arch = arch_model(residual_ARIMA,p = 1,q = 1) 
res_arch = arch.fit(update_freq=5)
print(res_arch.summary())
res_arch.plot(annualize='D')


innovations = residual_ARIMA/res_arch.conditional_volatility

params_norm = st.norm.fit(innovations)
params_t = st.t.fit(innovations)


plt.plot(residual_ARIMA/res_arch.conditional_volatility,color='blue')
plt.show()
print(params_norm)

