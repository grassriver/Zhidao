#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
n_samples = 10000

z = np.random.normal(size=n_samples)
x = np.ones((n_samples,))
x[0] = 0
s = np.ones((n_samples,))

m,kappa, theta = [0.05,-0.3,0.2]
a,beta, gamma = [0.01,0.95,0.03]

# Simulate ARMA
for t in range(n_samples):
    
    x[t] = m + kappa*x[t-1]+theta*z[t-1]+z[t]

print(np.round([m/(1-kappa), np.mean(x),(1+2*theta*kappa+theta*theta)/(1-kappa*kappa),np.var(x)],3))


for t in range(n_samples):
    s[t] = np.sqrt(a + beta*s[t-1]*s[t-1] + gamma*np.square(s[t-1]*z[t-1]))
    x[t] = m + kappa*x[t-1] + theta*z[t-1]*s[t-1]+z[t]*s[t]

#ts_data = x
#plt.plot(ts_data)
#plt.show()

# compare with theoritical mean and sigma

print(np.round([[m/(1-kappa), np.mean(x)],[(1+2*theta*kappa+theta*theta)/(1-kappa*kappa)*a/(1-beta-gamma),np.var(x)]],2))

# Fit with ARMA-GARCH model

# ARMA
from statsmodels.tsa.arima_model import ARIMA
model_arma = ARIMA(ts_data, order=(1, 0, 1))  
results_arma = model_arma.fit(disp=-1)  

residule_arma = results_arma.resid

print(results_arma.summary())

# GARCH
from arch import arch_model

arch = arch_model(residule_arma,p = 1,q = 1) 
res_arch = arch.fit(update_freq=5)
print(res_arch.summary())


# change to normal innovations
    

np.random.seed(1)
n_samples = 10000

# Use student t
z = np.random.standard_t(20,size=n_samples)
x = np.ones((n_samples,))
x[0] = 0
s = np.ones((n_samples,))

m,kappa, theta = [0.05,-0.3,0.2]
a,beta, gamma = [0.01,0.95,0.03]

for t in range(n_samples):
    s[t] = np.sqrt(a + beta*s[t-1]*s[t-1] + gamma*np.square(s[t-1]*z[t-1]))
    x[t] = m + kappa*x[t-1] + theta*z[t-1]*s[t-1]+z[t]*s[t]

print(np.round([[m/(1-kappa), np.mean(x)],[(1+2*theta*kappa+theta*theta)/(1-kappa*kappa)*a/(1-beta-gamma),np.var(x)]],2))

from statsmodels.tsa.arima_model import ARIMA
model_arma = ARIMA(ts_data, order=(1, 0, 1))  
results_arma = model_arma.fit(disp=-1)  

residule_arma = results_arma.resid

print(results_arma.summary())

# GARCH
from arch import arch_model

arch = arch_model(residule_arma,p = 1,q = 1,dist='StudentsT') 
res_arch = arch.fit(update_freq=5)
print(res_arch.summary())
