


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:23:36 2018

Statistical tool for time series data

@author: tianyulu
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf


def best_fit_distribution(data, bins=200, ax=None):
    """
    Model data by finding best fit distribution to data
    Currently support normal and student t distribution
    
    Parameters
    -----------
    data: {ndarray, Series}
            i.i.d random variable
    bins: integer, optional
            used in histogram
    ax: optional
    
    Returns
    ---------
    best_distribution: selected distribution
    best_params: parameters of selected distribution
    
    """
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [st.norm,st.t]
    
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




def test_stationarity(timeseries,plot = False, output = False):
    """
    Test the hypothesis of stationary by Dickey-Fuller test 

    Parameters
    -----------
    timeseries: {ndarray, Series}
            time series
    plot: bool, optional
            plot rolling statistics

    Returns
    ---------
    Results of Dickey-Fuller Test 
    
    """
    
    #Determing rolling statistics
    rolmean = pd.DataFrame.rolling(timeseries,window=12,center=False).mean()
    rolstd = pd.DataFrame.rolling(timeseries,window=12,center=False).std()
    
    #Plot rolling statistics:
    if plot==True:
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    if output==True:
        print('Results of Dickey-Fuller Test:')
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)

    
    return dfoutput

def normalize(x):
    """
    normalize data by substract mean and divided by standard deviation
    
    Parameters:
    ----------
    x: array like 1d
    
    Returns
    -----------
    x_norm: array like 1d

    """
    x_norma = (x-np.mean(x))/np.std(x)

def ljungbox_test(x, lags=2, boxpierce=False):
    """
    Parameters
    ----------
    x : array_like, 1d
        data series, regression residuals when used as diagnostic test
        lags : None, int or array_like
        If lags is an integer then this is taken to be the largest lag
        that is included, the test result is reported for all smaller lag length.
        If lags is a list or array, then all lags are included up to the largest
        lag in the list, however only the tests for the lags in the list are
        reported.
        If lags is None, then the default maxlag is 12*(nobs/100)^{1/4}
        boxpierce : {False, True}
        If true, then additional to the results of the Ljung-Box test also the
        Box-Pierce test results are returned

    Returns
    -------
        lbvalue : float or array
        test statistic
        pvalue : float or array
        p-value based on chi-square distribution
        bpvalue : (optionsal), float or array
        test statistic for Box-Pierce test
        bppvalue : (optional), float or array
        p-value based for Box-Pierce test on chi-square distribution

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is reported to have better
    small sample properties.
    """

            
    if boxpierce==False:
         lbvalue,pvalue = acorr_ljungbox(x,lags=lags, boxpierce=False)
         print('Ljung-Box test results\n p-value: {0} \n test statistics: {1}'.format
               (np.round(lbvalue,4),np.round(pvalue,4)))
         return lbvalue, pvalue
    else:
         lbvalue,pvalue,bpvalue,bppvalue = acorr_ljungbox(x,lags=lags, boxpierce=True)
         print('Ljung-Box test results\n p-value: {0} \n test statistics: {1}'.format
               (np.round(lbvalue,4),np.round(pvalue,4)))
         print('Box-Pierce test results\n p-value: {0} \n test statistics: {1}'.format
               (np.round(bpvalue,4),np.round(bppvalue,4)))
         return lbvalue,pvalue,bpvalue,bppvalue


    
    
def acf_pacf(ts_data, plot = 'acf_pacf'):
    """
    acf and pacf plot
    
    Parameters
    ----------
    ts_data: array_like, 1d
    plot: string, chosse from 'acf', 'pacf','acf_pacf'
    
    Returns:
    acf and/or pacf plot
    """
        
    lag_acf = acf(ts_data, nlags=20)
    lag_pacf = pacf(ts_data, nlags=20, method='ols')
    if plot in ['acf','acf_pacf']:
        #Plot ACF: 
        plt.subplot(121) 
        plt.plot(lag_acf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
        plt.title('Autocorrelation Function')
        plt.show()
        print(plot)
    elif plot in ['pacf','acf_pacf']:
        #Plot PACF:
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(ts_data)),linestyle='--',color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()
        plt.show()
        print(plot)
    
   
def kstest():

    print('scipy.stats.kstest')   
