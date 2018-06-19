#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:39:08 2018

@author: zifandeng
"""
def rolling_bus_day(window,t,d_offsets,method):
    date_range=[]
    date_temp3 = pd.to_datetime(t)
    date_range.append(t)
    for i in range(1,window):
        if method == 'backward':
            date_temp = (date_temp3-d_offsets*i)
        else:
            date_temp = (date_temp3+d_offsets*i)
        #date_temp2 =  bus_calender[bus_calender<=date_temp.strftime('%Y-%m-%d')].dropna().tail(1)   
        date_temp2 = date_temp.strftime('%Y-%m-%d')
        date_range.append(date_temp2)
        #date_temp3= pd.to_datetime(date_temp2.iloc[0,0])
    date_range = pd.to_datetime(date_range).sort_values()
    return date_range

def get_date_range(bus_calender,t,window,frequency,method = 'backward'):   
    # check frequency
    if frequency == 'annually':
        date_range = rolling_bus_day(window,t,pd.offsets.relativedelta(years=1),method)
    elif frequency == 'quarterly':
        date_range = rolling_bus_day(window,t,pd.offsets.relativedelta(months=3),method)
    elif frequency == 'monthly':
        date_range = rolling_bus_day(window,t,pd.offsets.relativedelta(months=1),method)
    elif frequency == 'weekly':
        date_range = rolling_bus_day(window,t,pd.offsets.relativedelta(weeks=1),method)
    elif frequency == 'daily':
        bday_offset = lambda n: pd.offsets.CustomBusinessDay(n, calendar=bus_calender)
        date_range = rolling_bus_day(window,t,bday_offset(1),method)
    else:
        raise ValueError('Please enter a valid frequency!')
    return date_range

def get_mkt_index(bus_calender,t,window,frequency,mkt):
    converter = {'daily':1,'weekly':5,'monthly':21,'quarterly':63,'annually':252}
    date_range = get_date_range(bus_calender,t,window*converter[frequency],'daily')
    begin = date_range[0]
    mkt = mkt.sort_values(by = 'date')
    mkt['date'] = pd.to_datetime(mkt['date'])
    mkt = mkt.set_index('date')
    mkt['mkt_ret'] = np.log(mkt['close'])-np.log(mkt['close'].shift(1))
    mkt = mkt[date_range[1].strftime('%Y-%m-%d'):t]
    mkt = mkt.dropna()
    
    mkt2 = convert_return(mkt,frequency)
   # for i in range(1,len(date_range)):
   #     mkt.loc[(mkt.index>date_range[i-1]) & (mkt.index<=date_range[i]),'group']=date_range[i]        
   # mkt2 = pd.DataFrame(mkt.groupby('group')['mkt_ret'].sum())
    return mkt2

def capm_modeling(t,frequency,window,stk_list = None,riskfree=0.0,mkt = None,
                  stocks_price_old=None, business_calendar=None, industry=None):
    
    """
    Stand at time t,
    Run the CAPM model for a rolling period before time t (including t);
    
    Parameters
    ----------
    conn: 
        sqlite3 connection;
    t: 
        string; a string of time in form of 'yyyy-mm-dd';
    frequency:
        string; a string of frequency for returns 
                i.e., 'daily','weekly','monthly','quarterly','annually';
    code_list:
        list; a list for stock code, default None;
              if None, then the entire pool of stock as of time t would be considered
    
    riskfree:
        float; a floating number for risk free rate, default 0.0
    
    window:
        int; an integer for rolling window input, in the unite of frequency
             i.e., if window = 2, frequency = 'weekly', it means 2 weeks
    mkt_code:
        string; a string for market index, default 'sh000300'

    Returns
    -------
    coeficient matrices: pd.DataFrame
        the matrix containing the information of 
        [code, available beginning date, end date, number of observation, alpha, beta, r_squared] 
    """  
    # Get Business Calender to check if selected date is business day
    converter = {'daily':1,'weekly':5,'monthly':21,'quarterly':63,'annually':252}
    #since = time.time()
    date_range = get_date_range(business_calendar,t,window*converter[frequency],'daily')
    begin = date_range[0]
    stk_list = pd.DataFrame(stk_list)
    stk_list.columns = ['code']
    data = stocks_price_old[stk_list['code']]
    data = data[date_range[1].strftime('%Y-%m-%d'):t]
    # get the time period before and including t
    mkt = get_mkt_index(business_calendar,t,window,frequency,mkt)[['mkt_ret']] 
    # Extract all stocks as of time t
#    for i in range(1,len(date_range)):
#        data.loc[(data.index>date_range[i-1]) & (data.index<=date_range[i]),'group']=date_range[i]             
#    data2 = pd.DataFrame(data.groupby('group').agg(lambda x:x.sum(skipna = False)))
#    data2 = data2.sort_index()
    # Calculate coeficients
    data2 = convert_return(data,frequency)
    results = (map(functools.partial(get_coef,data2,mkt,riskfree),stk_list['code']))
    model_coef = list(results)
    #time_lag = time.time()-since        
    cols = ['code','begin','end','number of obs','alpha','beta','r2']
    model_coef = pd.DataFrame(model_coef,columns = cols)
    #print('Coefficient calculation completed in {:.0f}m {:.0f}s'.format(time_lag // 60, time_lag % 60))
    
    #industry_query = 'select code,industry from stock_basics'
    model_coef = pd.merge(model_coef,industry,left_on = 'code',right_on = 'code',how = 'left')
    # return
    model_coef_sub = model_coef[model_coef['number of obs']>50]
    industry_avg_beta = pd.DataFrame(model_coef_sub.groupby('industry')['beta'].apply(np.nanmean))
    industry_avg_beta.columns = ['industry_avg_beta']
    industry_avg_alpha = pd.DataFrame(model_coef_sub.groupby('industry')['alpha'].apply(np.nanmean))
    industry_avg_alpha.columns = ['industry_avg_alpha']
    model_coef = pd.merge(model_coef,industry_avg_beta,left_on = 'industry',right_index = True,how = 'left')
    model_coef = pd.merge(model_coef,industry_avg_alpha,left_on = 'industry',right_index = True,how = 'left')
    model_coef.loc[model_coef['number of obs']<np.ceil(window*0.2),'beta'] = model_coef[model_coef['number of obs']<np.ceil(window*0.2)]['industry_avg_beta']
    model_coef.loc[model_coef['number of obs']<np.ceil(window*0.2),'alpha'] = model_coef[model_coef['number of obs']<np.ceil(window*0.2)]['industry_avg_alpha']
    return model_coef

def convert_return(data,frequency):
    if frequency == 'annually':
        data2 =  data.resample('A',kind='period').sum()
    elif frequency == 'quarterly':
        data2 =  data.resample('Q',kind='period').sum()
    elif frequency == 'monthly':
        data2 =  data.resample('M',kind='period').sum()
    elif frequency == 'weekly':
        data2 =  data.resample('W',kind='period').sum()
    elif frequency == 'daily':
        data2 = data
    else:
        raise ValueError('Please enter a valid frequency!')
    return data2

def get_coef(stk,mkt,riskfree,code):
    try:
        data = stk[[code]]
        data = data.dropna()
        data = pd.merge(data,mkt,'left',left_index= True,right_index = True)
        beta = ratio.get_beta(data[code],data['mkt_ret'],riskfree)
        alpha = ratio.get_annulized_alpha(asset=data[code],market=data['mkt_ret'],annualization=1,riskfree=riskfree)
        r2 = ratio.get_rsquare(asset=data[code],market=data['mkt_ret'],riskfree=riskfree)
        begin = data.index[0]
        end = data.index[~0]
        n= len(data[code])
    except:
        print('No Return Data Available for '+ code)
        beta = np.nan
        alpha = np.nan
        r2 = np.nan
        begin = np.nan
        end = np.nan 
        n = 0
    return code,begin,end,n,alpha,beta,r2