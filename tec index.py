
import pandas as pd
import sqlite3 as sql
import sys
sys.path.append('/Users/yunongwu/Documents/Portfolio_Intelligence/Code/')
conn = sql.connect('/Users/yunongwu/Nustore Files/PI/data/data.db')
c = conn.cursor()
c.execute("select * from stocks_price")

price = pd.DataFrame(c.fetchall())
price.columns =['Date','Open','Close','High','Low','Volume','Code']
# a=price.loc[(price['Code']=='000517')|(price['Code']=='600399')]
b=price.sort_values(by=['Code','Date'])
b['Date'] = pd.DatetimeIndex(b['Date'])
b = b.set_index('Date')

def rsi6(ClosePrice,n=6):
    ClosePrice = pd.DataFrame(ClosePrice)
    change=(ClosePrice-ClosePrice.shift(1)).fillna(0)
    up, down=change.copy(),change.copy()
    up[up<0]=0
    down[down>0]=0
    RolUp=up.ewm(ignore_na=False,adjust=True,alpha=1/n,min_periods=0).mean()
    RolDown=down.ewm(ignore_na=False,adjust=True,alpha=1/n,min_periods=0).mean().abs()
    RS=RolUp/RolDown
    rsi=100-(100/(1+RS))
    return round(rsi,2)

rsi6=b.groupby('Code')[['Close']].apply(rsi6).reset_index()
rsi6=rsi6.rename(columns={'Close':'RSI6'})

def rsi12(ClosePrice,n=12):
    ClosePrice = pd.DataFrame(ClosePrice)
    change=(ClosePrice-ClosePrice.shift(1)).fillna(0)
    up, down=change.copy(),change.copy()
    up[up<0]=0
    down[down>0]=0
    RolUp=up.ewm(ignore_na=False,adjust=True,alpha=1/n,min_periods=0).mean()
    RolDown=down.ewm(ignore_na=False,adjust=True,alpha=1/n,min_periods=0).mean().abs()
    RS=RolUp/RolDown
    rsi=100-(100/(1+RS))
    return round(rsi,2)

rsi12=b.groupby('Code')[['Close']].apply(rsi12).reset_index()
rsi12=rsi12.rename(columns={'Close':'RSI12'})

def rsi24(ClosePrice,n=24):
    ClosePrice = pd.DataFrame(ClosePrice)
    change=(ClosePrice-ClosePrice.shift(1)).fillna(0)
    up, down=change.copy(),change.copy()
    up[up<0]=0
    down[down>0]=0
    RolUp=up.ewm(ignore_na=False,adjust=True,alpha=1/n,min_periods=0).mean()
    RolDown=down.ewm(ignore_na=False,adjust=True,alpha=1/n,min_periods=0).mean().abs()
    RS=RolUp/RolDown
    rsi=100-(100/(1+RS))
    return round(rsi,2)

rsi24=b.groupby('Code')[['Close']].apply(rsi24).reset_index()
rsi24=rsi24.rename(columns={'Close':'RSI24'})

#Simple Moving Average
def sma5(ClosePrice,n=5):
    ClosePrice=pd.DataFrame(ClosePrice)
    ma=ClosePrice.copy()
    ma5=ma.rolling(window=n,center=False).mean()
    return round(ma5,2)

sma5=b.groupby('Code')[['Close']].apply(sma5).reset_index()
sma5=sma5.rename(columns={'Close':'SMA5'})

def sma10(ClosePrice,n=10):
    ClosePrice=pd.DataFrame(ClosePrice)
    ma=ClosePrice.copy()
    ma10=ma.rolling(window=n,center=False).mean()
    return round(ma10,2)

sma10=b.groupby('Code')[['Close']].apply(sma10).reset_index()
sma10=sma10.rename(columns={'Close':'SMA10'})

def sma20(ClosePrice,n=20):
    ClosePrice=pd.DataFrame(ClosePrice)
    ma=ClosePrice.copy()
    ma20=ma.rolling(window=n,center=False).mean()
    return round(ma20,2)

sma20=b.groupby('Code')[['Close']].apply(sma20).reset_index()
sma20=sma20.rename(columns={'Close':'SMA20'})

def sma30(ClosePrice,n=30):
    ClosePrice=pd.DataFrame(ClosePrice)
    ma=ClosePrice.copy()
    ma30=ma.rolling(window=n,center=False).mean()
    return round(ma30,2)

sma30=b.groupby('Code')[['Close']].apply(sma30).reset_index()
sma30=sma30.rename(columns={'Close':'SMA30'})

def sma60(ClosePrice,n=30):
    ClosePrice=pd.DataFrame(ClosePrice)
    ma=ClosePrice.copy()
    ma60=ma.rolling(window=n,center=False).mean()
    return round(ma60,2)

sma60=b.groupby('Code')[['Close']].apply(sma60).reset_index()
sma60=sma60.rename(columns={'Close':'SMA60'})

#High/Low
def pre20high(ClosePrice,n=20):
    ClosePrice=pd.DataFrame(ClosePrice)
    pre=ClosePrice.copy()
    pre20high=pre.rolling(window=n).max()
    return round(pre20high,2)

pre20high=b.groupby('Code')[['Close']].apply(pre20high).reset_index()
pre20high=pre20high.rename(columns={'Close':'20HIGH'})

def pre20low(ClosePrice,n=20):
    ClosePrice=pd.DataFrame(ClosePrice)
    pre=ClosePrice.copy()
    pre20low=pre.rolling(window=n).min()
    return round(pre20low,2)

pre20low=b.groupby('Code')[['Close']].apply(pre20low).reset_index()
pre20low=pre20low.rename(columns={'Close':'20LOW'})

def pre50high(ClosePrice,n=50):
    ClosePrice=pd.DataFrame(ClosePrice)
    pre=ClosePrice.copy()
    pre50high=pre.rolling(window=n).max()
    return round(pre50high,2)

pre50high=b.groupby('Code')[['Close']].apply(pre50high).reset_index()
pre50high=pre50high.rename(columns={'Close':'50HIGH'})

def pre50low(ClosePrice,n=50):
    ClosePrice=pd.DataFrame(ClosePrice)
    pre=ClosePrice.copy()
    pre50low=pre.rolling(window=n).min()
    return round(pre50low,2)

pre50low=b.groupby('Code')[['Close']].apply(pre50low).reset_index()
pre50low=pre50low.rename(columns={'Close':'50LOW'})

#Gap
def gap(ClosePrice):
    ClosePrice = pd.DataFrame(ClosePrice)
    gap = ClosePrice['Open']-ClosePrice['Close'].shift(1)  
    gap = pd.DataFrame({'GAP':gap})
    gap = gap.copy()  
    return round(gap,2)
gap=b.groupby('Code').apply(gap).reset_index()

#Close Price Change
def closechange(ClosePrice):
    ClosePrice = pd.DataFrame(ClosePrice)
    change = ClosePrice['Close']-ClosePrice['Close'].shift(1)  
    change = pd.DataFrame({'CloseChange':change})
    change = change.copy()  
    return round(change,2)
closechange=b.groupby('Code').apply(closechange).reset_index()

#Change from Open
def changefromopen(ClosePrice):
    ClosePrice = pd.DataFrame(ClosePrice)
    change = ClosePrice['Close']-ClosePrice['Open']
    change = pd.DataFrame({'ChangeFromOpen':change})
    change = change.copy()  
    return round(change,2)
changefromopen=b.groupby('Code').apply(changefromopen).reset_index()

#Amplitude
def amplitude(ClosePrice):
    ClosePrice = pd.DataFrame(ClosePrice)
    amplitude = round(100*(ClosePrice['High']-ClosePrice['Low'])/ClosePrice['Close'].shift(1),4)
    amplitude = pd.DataFrame({'Amplitude%':amplitude})
    amplitude = amplitude.copy()
    return amplitude

amplitude=b.groupby('Code').apply(amplitude).reset_index()

#HighLowRange
def HighLowRange(ClosePrice):
    ClosePrice = pd.DataFrame(ClosePrice)
    ClosePrice['Close_Shift'] = ClosePrice.groupby('Code').shift(1)['Close']
    ClosePrice['High_Range%'] = 100*round((ClosePrice['Close_Shift']-ClosePrice['High']).abs()/ClosePrice['Close_Shift'],4)
    ClosePrice['Low_Range%'] = 100*round((ClosePrice['Close_Shift']-ClosePrice['Low']).abs()/ClosePrice['Close_Shift'],4)    
    return ClosePrice[['Code', 'High_Range%', 'Low_Range%']]

highlowrange=HighLowRange(b).reset_index()

highlowrange.loc[(highlowrange['High_Range%'] > 11) | highlowrange['Low_Range%']>11]

#MACD
def MACD(ClosePrice, nslow=26, nfast=12, nma=9):
    ClosePrice = pd.DataFrame(ClosePrice)  
    emaslow = ClosePrice["Close"].ewm(ignore_na=False,span=nslow, min_periods=0,adjust=True).mean()
    emafast = ClosePrice["Close"].ewm(ignore_na=False,span=nfast, min_periods=0,adjust=True).mean()
    diff = emafast - emaslow
    dea = diff.ewm(ignore_na=False,span=nma, min_periods=0,adjust=True).mean()
    macd = 2*(diff - dea)
    result = pd.DataFrame({'MACD_DIFF': round(emafast-emaslow,2),'MACD_DEA':round(dea,2),'MACD':round(macd,2)})
    result = result.copy()
    return result

MACD=b.groupby('Code')[['Close']].apply(MACD).reset_index()


dfs = [rsi6,rsi12,rsi24,sma5,sma10,sma20,sma30,sma60,pre20high,pre20low,pre50high,pre50low,gap,closechange,changefromopen,amplitude,highlowrange,MACD]
from functools import reduce
df_final = reduce(lambda left,right: pd.merge(left,right,on=['Code','Date']), dfs)

df_final.to_sql(con=conn, name='technical', if_exists='replace')
conn.commit()

conn.close()