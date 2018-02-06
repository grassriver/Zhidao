import pandas as pd
import scipy.stats as st

# Calculate historical VaR. The input of the function is 'select' and 'cutoff'.
# 'select' is a series of returns. 'cutoff' is the significance level.
# 'interpolation': when the desired quantile lies between two data points i and j, 'lower' method uses i.
def hist_var(select,cutoff=0.01):
    return round(select.quantile(cutoff,interpolation='lower'),4)
    
# Calculate normal VaR. The delta-normal approach assumes the returns follows a normal distribution.
# VaR(a%) = u - sigma * Za
def delta_normal_var(select,cutoff=0.01):
    return round(select.mean()-select.std()*st.norm.ppf(1-cutoff),4)
    

# Volatility-weighted historical simulation method.
# 'span' specifies decay in terms of span, a=2/(span+1), 'a' is the decay parameter.
def ewma_var(select,cutoff=0.01,span_value=1.05):
    adj_returns=pd.ewma(select,span=span_value)
    return round(adj_returns.quantile(cutoff,interpolation='lower'),4)
    
# Calculates conditional VaR (a.k.a. expected shortfall). It is viewed as an average of all losses greater than the VaR.   
def conditional_var(select,cutoff=0.01):
    var = select.quantile(cutoff)
    return  round(select[select <= var].mean(),4)

