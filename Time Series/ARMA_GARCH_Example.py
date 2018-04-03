#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 23:32:01 2018

@author: tianyulu
"""

from arma_garch import ARMA_GARCH

model = ARMA_GARCH(order1=[1,0,1],order2=[1,1])

test_data = model.simulation(1000,[0.05,0.5,0.2],[0.01,0.02,0.95])

model.estimation(test_data)

model.prediction(10)


