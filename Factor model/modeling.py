#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:54:35 2018

@author: zifandeng
"""

from sklearn import linear_model as lm
from sklearn import metrics
import statsmodels.api as sm
import pandas as pd
import numpy as np

#%% Build model
def build_model(df,mod,x_var,y_var):
    var_list = x_var.copy()
    var_list.insert(0,y_var)
    subset = df[var_list]
    subset = subset.dropna()
    x = subset[x_var]
    y = subset[y_var]
    fit1 = mod.fit(x,y)
    return mod.coef_,mod.intercept_,fit1.score(x,y)



    