import pandas as pd

def hist_var(select,cutoff=0.01):
    return select.quantile(cutoff,interpolation='lower')
    

def delta_normal_var(select,cutoff=0.01):
    return select.mean()-select.std()*st.norm.ppf(1-cutoff)
    


def ewma_var(select,cutoff=0.01,span_value=1.05):
    adj_returns=pd.ewma(select,span=span_value)
    return adj_returns.quantile(cutoff,interpolation='lower')
    

    
def cvar(select,cutoff=0.01):
    var = select.quantile(cutoff)
    return  select[select <= var].mean()

