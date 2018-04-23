# -*- coding: utf-8 -*-
"""
dataclass for optimization

Created on Sun Apr 22 12:18:14 2018

@author: Kai Zheng
"""

import pandas as pd
import general_tools as tool

class method(object):
    def __init__(self, name, method, kwargs):
        self.name = name
        self.method = method
        self.kwargs = kwargs
    
    def run(self):
        return self.method(**self.kwargs)
    
class mu_method(method):
    def update(self, **kwargs):
        self.kwargs.update(kwargs)
        self.kwargs['start'] = tool.last_trading_day(self.kwargs['conn'], 
                   self.kwargs['start'], 
                   self.kwargs['business_calendar'])
    
class sigma_method(method):
    def update(self, **kwargs):
        self.kwargs.update(kwargs)
        self.kwargs['start'] = tool.last_trading_day(self.kwargs['conn'], 
                   self.kwargs['start'], 
                   self.kwargs['business_calendar'])
    
class opt_method(method):
    def update(self, **kwargs):
        self.kwargs.update(kwargs)