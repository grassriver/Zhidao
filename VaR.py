import pandas as pd
import scipy.stats as st

def hist_var(select,cutoff=0.01):
    """
    Calculates historical Value at Risk (VaR).

    Parameters
    ----------
    select : pandas.DataFrame daily log returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.01.

    Returns
    -------
    VaR : float
        The VaR value.
    """
    return round(select.quantile(cutoff,interpolation='lower'),4)
    

def delta_normal_var(select,cutoff=0.01):
    """
    Calculates normal VaR. The delta-normal approach assumes the returns follows a normal distribution.

    Parameters
    ----------
    select : pandas.DataFrame daily log returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.01.

    Returns
    -------
    Normal VaR : float
        The normal VaR value. 
    """
    return round(select.mean()-select.std()*st.norm.ppf(1-cutoff),4)
    

def ewma_var(select,cutoff=0.01,span_value=1.05):
    """
    Calculates EWMA VaR. This is a volatility-weighted historical simulation method.

    Parameters
    ----------
    select : pandas.DataFrame daily log returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.01.
    span_value : specifies decay in terms of span, a=2/(span+1), 'a' is the decay parameter.

    Returns
    -------
    EWMA VaR : float
        EWMA VaR value.
    """
    adj_returns=select.ewm(span=span_value).mean()
    return round(adj_returns.quantile(cutoff,interpolation='lower'),4)
    
  
def conditional_var(select,cutoff=0.01):
    """
    Calculates conditional VaR (a.k.a. expected shortfall). It is viewed as an average of 
        all losses greater than the VaR. 

    Parameters
    ----------
    select : pandas.DataFrame daily log returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.01.

    Returns
    -------
    CVaR : float
        The CVaR value.
    """
    var = select.quantile(cutoff)
    return  round(select[select <= var].mean(),4)

