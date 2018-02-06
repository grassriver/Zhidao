import pandas as pd

def weekly_return(Close):
    Close=Close.reset_index()
    datetime=pd.to_datetime(Close['date'])
    weekday=datetime.dt.weekday_name
    Close=Close.assign(weekday=weekday.values)
    Close.set_index('date',inplace=True)
    Close=Close[Close['weekday']=='Friday']
    Close=Close.drop('weekday',axis=1)
    weekly_returns= round(Close / Close.shift(1)-1,4)   
    return weekly_returns.iloc[1:]
    
def monthly_return(Close):
    Close.index=Close.index.to_period('M')
    monthly_Close=Close.groupby(Close.index).last()
    monthly_returns= round(monthly_Close / monthly_Close.shift(1)-1,4)
    return monthly_returns.iloc[1:]