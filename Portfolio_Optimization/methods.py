# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:35:00 2018

@author: Kai Zheng
"""
#%%
# =============================================================================
# mu_method = mu_hist
#             mu_capm
#             mu_reverse_capm
# 
# sigma_method = sigma_hist
#                sigma_barra
# 
# opt_method = opt_quadratic
#              opt_s
#              opt_v
#              opt_restricted_s
#              opt_restricted_v 
# 
# =============================================================================

from Portfolio_Optimization.methodclass import method
import Portfolio_Optimization.mv as mv

#============mu_capm=====================
kwargs={'conn': conn, 
        'start': (pd.to_datetime(start)-pd.Timedelta(1)).strftime('%Y-%m-%d'), 
        'lookback_win': 252, 
        'stk_list': code_list,
        'proj_period':30, 
        'proj_method':'arma_garch',
        'freq': 'daily'}
mu_capm = method('capm', capm.capm_mu, kwargs)

#============mu_hist======================
kwargs={'conn':conn, 
        'code_list':code_list,
        'start':(start-(end-start)).strftime('%Y-%m-%d'),
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d')}
mu_hist = method('hist', mv.hist_expect_mu, kwargs)


#============sigma_hist====================
kwargs={'conn':conn, 
        'code_list':code_list,
        'start':(start-(end-start)).strftime('%Y-%m-%d'),
        'end':(start-pd.Timedelta(1)).strftime('%Y-%m-%d')}
sigma_hist = method('hist', mv.hist_expect_sigma, kwargs)


#========opt_quadratic==============
opt_method = method('opt_quadratic', method=mv.opt_quadratic, 
                    kwargs={'mu':mu, 'sigma':sigma, 'l':10})


#==========opt_s====================
opt_method  = method('opt_s', method=mv.opt_s, 
                    kwargs={'mu':mu, 'sigma':sigma})

#==========opt_v====================
opt_method  = method('opt_v', method=mv.opt_v, 
                    kwargs={'mu':mu, 'sigma':sigma})


#==========sigma barra==============
kwargs={'code_list':code_list,
        'start':start,
        'window':window,
        'model_data':model_data,
        'wls_resid':wls_resid,
        'factor_list2':factor_list2,
        'conn':data_path}
sigma_barra = sigma_method('sigma_fm', FM_opt.barra_stk_cov, kwargs)

