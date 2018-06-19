#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:06:28 2018

@author: zifandeng
"""
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
        
def rv_generator(distri,paras,n):
    rv = {'norm': lambda paras,n:stats.norm.rvs(paras[0],paras[1],size = n),
           'gamma': lambda paras,n:stats.gamma.rvs(paras[0],paras[1],paras[2],size = n),
           'beta': lambda paras,n:stats.beta.rvs(paras[0],paras[1],paras[2],paras[3],size = n),
           't':lambda paras,n:stats.t.rvs(paras[0],paras[1],paras[2],size = n),
           'cauchy':lambda paras,n:stats.cauchy.rvs(paras[0],paras[1],size = n),
           'lognorm':lambda paras,n:stats.lognorm.rvs(paras[0],paras[1],paras[2],size = n),
           'gumbel':lambda paras,n:stats.gumbel_r.rvs(paras[0],paras[1],size = n)}
    return rv[distri](paras,n)

def get_ppf(x,distri,paras):
    ppf =  {'norm': lambda x,paras:stats.norm.ppf(x,paras[0],paras[1]),
           'gamma': lambda x,paras:stats.gamma.ppf(x,paras[0],paras[1],paras[2]),
           'beta': lambda x,paras:stats.beta.ppf(x,paras[0],paras[1],paras[2],paras[3]),
           't':lambda x,paras:stats.t.ppf(x,paras[0],paras[1],paras[2]),
           'cauchy':lambda x,paras:stats.cauchy.ppf(x,paras[0],paras[1]),
           'lognorm':lambda x,paras:stats.lognorm.ppf(x,paras[0],paras[1],paras[2]),
           'gumbel':lambda x:stats.gumbel_r.ppf(x,paras[0],paras[1])}
    return ppf[distri](x,paras)

def get_pdf(x,distri,paras):
    pdf = {'norm': lambda x,paras:stats.norm.pdf(x,paras[0],paras[1]),
           'gamma': lambda x,paras:stats.gamma.pdf(x,paras[0],paras[1],paras[2]),
           'beta': lambda x,paras:stats.beta.pdf(x,paras[0],paras[1],paras[2],paras[3]),
           't':lambda x,paras:stats.t.pdf(x,paras[0],paras[1],paras[2]),
           'cauchy':lambda x,paras:stats.cauchy.pdf(x,paras[0],paras[1]),
           'lognorm':lambda x,paras:stats.lognorm.pdf(x,paras[0],paras[1],paras[2]),
           'gumbel':lambda x:stats.gumbel_r.ppf(x,paras[0],paras[1])}   
    return pdf[distri](x,paras)

def get_cdf(x,distri,paras):
    cdf = {'norm': lambda x,paras:stats.norm.cdf(x,paras[0],paras[1]),
           'gamma': lambda x,paras:stats.gamma.cdf(x,paras[0],paras[1],paras[2]),
           'beta': lambda x,paras:stats.beta.cdf(x,paras[0],paras[1],paras[2],paras[3]),
           't':lambda x,paras:stats.t.cdf(x,paras[0],paras[1],paras[2]),
           'cauchy':lambda x,paras:stats.cauchy.cdf(x,paras[0],paras[1]),
           'lognorm':lambda x,paras:stats.lognorm.cdf(x,paras[0],paras[1],paras[2]),
           'gumbel':lambda x:stats.gumbel_r.ppf(x,paras[0],paras[1])}   
    return cdf[distri](x,paras)

def distribution_fit(x,distri,auto=False):
    percentile = [0.001,0.25,0.5,0.75,0.999]
    percentile2 = [0.1,25,50,75,99.9]
    score_func = lambda emp,theo:1/2*sum(np.power((emp-theo),2))
    score_record = 10000
    distri_record = 'norm'
    
    mthd = {'norm': lambda x:stats.norm.fit(x),
           'gamma': lambda x:stats.gamma.fit(x),
           'beta': lambda x:stats.beta.fit(x),
           't':lambda x:stats.t.fit(x),
           'cauchy':lambda x:stats.cauchy.fit(x),
           'lognorm':lambda x:stats.lognorm.fit(x)}
    
    columns = {'norm': ['loc','scale'],
           'gamma': ['a','loc','scale'],
           'beta': ['a','b','loc','scale'],
           't':['df','loc','scale'],
           'cauchy':['loc','scale'],
           'lognorm':['s','loc','scale']}
    if auto == True:
        for d in mthd.keys():
            paras = mthd[d](x)
            [t,p_val]=stats.kstest(x,d,paras)
            if p_val<=0.05:
                print(d+ 'fit does not pass the KS-test')
                continue
            emp = np.percentile(x,percentile2)
            theo = get_ppf(percentile,d,paras)
            score =score_func(emp,theo)
            if score < score_record:
                distri_record = d
                score_record = score
        distri = distri_record
        print('selected distribution is '+ distri)

    # get parameters
    paras = mthd[distri](x)
    # get KS test result
    [t,p_val]=stats.kstest(x,distri,paras)
    # Calculate percentile squared error
    emp = np.percentile(x,percentile2)
    theo = get_ppf(percentile,distri,paras)
    score =t            
    # plot
    x2 = np.linspace(get_ppf(0.001,distri,paras),get_ppf(0.999,distri,paras),1000)
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(211)
    ax.plot(x2, get_pdf(x2,distri,paras),'-', lw=2, alpha=0.6, label='Fitted Distribution')
    ax.hist(x,alpha=0.2,histtype='stepfilled',density=True)
    ax.legend(loc='best', frameon=False)
    ax2 = fig.add_subplot(212)
    ax2.plot(x2, get_cdf(x2,distri,paras),'-', lw=2, alpha=0.6, label='Fitted Distribution')
    ax2.plot(theo,percentile,'.', lw=10, alpha=0.6, label='Percentile')
    ax2.plot(np.percentile(x,np.linspace(0.001,99.9,50)),np.linspace(0.001,99.9,50)/100,'-', lw=2, alpha=0.6, label='Empirical Distribution')
    ax2.plot(emp,percentile,'.', lw=20, alpha=0.6, label='Percentile')
    ax2.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()
    # make it a dictionary
    paras = dict(zip(columns[distri],mthd[distri](x)))
    return [paras,p_val,score]

def simulation(n,marginal,copula,mu,cov,dof = None):
    cov = np.mat(cov)
    m = len(cov)
    A = np.linalg.cholesky(cov)
    diag = np.sqrt((np.diag(cov)))
    z = np.mat(rv_generator('norm',(0,1),(n,m)))
    x = []
    if copula == 'norm':
        for i in range(n):
            z_temp = np.transpose(z[i])
            Y=(A*z_temp)
            u = get_cdf(np.diag(Y/(diag)),'norm',(0,1))
            x.insert(i,[get_ppf(u[i],marginal,(mu[i],diag[i])) for i in range(0,m)])
    elif copula == 't':
        if dof == None:
            raise ValueError('No degree of freedom specified!')
        for i in range(n):
            z_temp = np.transpose(z[i])
            Y=(A*z_temp)
            s = stats.chi2.rvs(dof)
            w = np.sqrt(dof/s)*Y
            u = get_cdf(np.diag(w/(diag)),'t',(dof,0,1))
            x.insert(i,[get_ppf(u[i],marginal,(mu[i],diag[i])) for i in range(0,m)])
    x = pd.DataFrame(x)
    return x

def simulation_plot(x,marginal,mu,cov,plot_type='marginal'):    
    cov = np.mat(cov)
    diag = np.sqrt((np.diag(cov)))    
    #plot
    n = len(cov)
    n1 = np.ceil(np.sqrt(n))
    if plot_type == 'marginal':
        fig = plt.figure(figsize = (15,12))
        for i in range(1,n+1):
            ax = fig.add_subplot(n1,n1,i)
            ax.hist(np.array(x[i-1]),density=True)
            x2 = np.linspace(get_ppf(0.001,marginal,(mu[i-1],diag[i-1])),get_ppf(0.999,marginal,(mu[i-1],diag[i-1])),1000)
            ax.plot(x2, get_pdf(x2,marginal,(mu[i-1],diag[i-1])),'-', lw=2, alpha=0.6,label=marginal)
            ax.legend(loc='upper right', frameon=False)
    else:
        n1 = np.ceil(np.sqrt(np.math.factorial(n-1)))
        fig = plt.figure(figsize = (15,12))
        n2 = 0
        for i in range(1,n+1):
            for j in range(i,n):
                ax = fig.add_subplot(n1,n1,1+n2)
                ax.scatter(x[i-1],x[j])
                n2 = n2+1
        plt.show()
    return fig