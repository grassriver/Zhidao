#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMA - GARCH model simulation, estimation and forecasting

Example:
----------
from arma_garch import ARMA_GARCH

model = ARMA_GARCH(order1=[1,0,1],order2=[1,1])

test_data = model.simulation(1000,[0.05,0.5,0.2],[0.01,0.02,0.95])

model.estimation(test_data)

model.prediction(10)

Created on Sat Mar 17 15:13:47 2018

@author: tianyulu
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st

from statsmodels.tsa.arima_model import ARIMA, ARMA, ARMAResults
from arch import arch_model

class ARMA_GARCH(object):
    def __init__(self, order1=[1,0,1],order2=[1,1]):
        """
        initiate model with ARMA order (m,n), and GARCH order (p,q)
        """
        self.arma_order = order1
        self.garch_order = order2
        
        
    def estimation(self,ts_data):
        """
        estimate ARMA and GARCH model.
        the residules from ARMA are fed into GARCH
        """
        arma_order = self.arma_order[:]
        model_arma = ARIMA(ts_data, arma_order)  
        results_arma = model_arma.fit(disp=-1)  

        residule_arma = results_arma.resid

        print(results_arma.summary())
        #res_arch.plot(annualize='D')
        p = self.garch_order[0]
        q = self.garch_order[1]
        arch = arch_model(residule_arma,p = p,q = q) 
        res_arch = arch.fit(update_freq=5)
        
        self.arma_params = results_arma.params
        self.garch_params = res_arch.params
        self.arma_fitted = results_arma.fittedvalues
        self.garch_fitted = res_arch.conditional_volatility
        
        print(res_arch.summary())
        
    
    def prediction(self,n_period):
        """
        Predict for next n periods
        """
        arma_params = self.arma_params
        garch_params = np.array(self.garch_params)
        
        

        
        m = self.arma_order[0]
        k = self.arma_order[1]
        n = self.arma_order[2]
        mu = arma_params[0]
        kappa = arma_params[1:m+1]
        theta = arma_params[m+1:m+n+1]

        p = self.garch_order[0]
        q = self.garch_order[1]
        # sigma^2 = w + alpha*e^2(t-1)+beta*sigma^2(t-1)
        w = garch_params[1]
        alpha = garch_params[2:p+2]
        beta = garch_params[p+2:p+q+2]
        
        # set the first few element of x and sigma equal to long run averge
        lag_max = max(m,n,p,q)
        
        x_pre = self.arma_fitted[-max(m,n):]
        s_pre = self.garch_fitted[-max(p,q):]
        
        z = np.random.normal(size=n_period+lag_max)
        x = np.ones((n_period+lag_max,))     
        s = np.ones((n_period+lag_max,))        
        x[0:lag_max] = x_pre
        s[0:lag_max] = s_pre
    
        
        for t in range(lag_max,n_period+lag_max):
            t_m = lag_max-m
            t_n = lag_max-n     
            t_p = lag_max-p
            t_q = lag_max-q
            
            s[t] = np.sqrt(w + np.sum(alpha*np.square(s[t-p:t]*z[t-p:t])) 
                +np.sum(beta*s[t-q:t]*s[t-q:t]))
            x[t] = mu + np.sum(kappa*x[t-m:t]) + np.sum(theta*z[t-n:t]*s[t-n:t-t_n]) + z[t]*s[t]

#            s = np.append(s,np.sqrt(w + np.sum(alpha*np.square(s[t-t_p-p:t-t_p]*z[t-t_p-p:t-t_p])) 
#                +np.sum(beta*s[t-t_q-q:t-t_q]*s[t-t_q-q:t-t_q])))
#            x = np.append(x,mu + np.sum(kappa*x[t-t_m-m:t-t_m]) + 
#                          np.sum(theta*z[t-t_n-n:t-t_n]*s[t-t_n-n:t-t_n]) + z[t-lag_max]*s[t])        
#       
        
        plt.plot(x)
        plt.show()
        
        self.prediction_x = x[m:]
        self.prediction_conditional_vol = s[1:]
        
        
        
        return
        
        
        
    def simulation(self,nsample,arma_params,garch_params):
        """
        simulate ARMA and GARCH process with normal distribution
        """
        arma_params = np.array(arma_params)
        garch_params = np.array(garch_params)
        
        np.random.seed(1)
        n_samples = nsample
        
        z = np.random.normal(size=n_samples)
        x = np.ones((n_samples,))
        
        s = np.ones((n_samples,))

        m = self.arma_order[0]
        k = self.arma_order[1]
        n = self.arma_order[2]
        mu = arma_params[0]
        kappa = arma_params[1:m+1]
        theta = arma_params[m+1:m+n+1]

        p = self.garch_order[0]
        q = self.garch_order[1]
        # sigma^2 = w + alpha*e^2(t-1)+beta*sigma^2(t-1)
        w = garch_params[0]
        alpha = garch_params[1:p+1]
        beta = garch_params[p+1:p+q+1]
        
        # set the first few element of x and sigma equal to long run averge
        x[0:m] = mu/(1-np.sum(kappa))
        s[0:q] = w/(1-alpha-beta)
        
        for t in range(m,n_samples):
            s[t] = np.sqrt(w + np.sum(alpha*np.square(s[t-p:t]*z[t-p:t]))+
             np.sum(beta*s[t-q:t]*s[t-q:t]))
            x[t] = mu + np.sum(kappa*x[t-m:t]) + np.sum(theta*z[t-n:t]*s[t-n:t]) + z[t]*s[t]

        
       
        ts_data = x
        plt.plot(ts_data)
        plt.show()
        
        # compare with theoritical mean and sigma
       
        print('mu , x_mean, sigma, x_std , s ')
        print(np.round([mu/(1-np.sum(kappa)), np.mean(x),np.sqrt((w/(1-alpha-beta))[0]),np.std(x),np.mean(s)],2))
        
        return ts_data
               

        
   