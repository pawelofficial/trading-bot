# this is algo which takes data and assigns it two metrics - winner looser boolean
# winner looser  score 
# both are based on tolerance t value which is basically a stop loss  or take profit percentage

import sys 
import numpy as np 
sys.path.append("../..")
from myfuns import myfuns as mf 
import pandas as pd 
import matplotlib.pyplot as plt 

# reading data fren 
def read_csv(path = None,filename=None ,scale = 5 ):
    if not path:
        path='../../data/'
    if not filename:
        filename='BTC-USD2022-01-10_2022-01-18'
    df=pd.read_csv(f'{path}/{filename}.csv', lineterminator=';')
    return mf.aggregate(df=df, scale=scale, src_col='timestamp')
# read winners or generatewinners 
if 1: # dump df to csv or read csv  
    df=read_csv()
    df.to_csv('winners.csv',index=False)
else:
    print('reading csv file')
    df=pd.read_csv('winners.csv')
    
    
def roi(base_row : pd.Series, row: pd.Series, mode: str ):
    if mode =='long':
        return row['close']/base_row['high']
    elif mode =='short':
        return row['close']/base_row['low']
    
# this should be split for longs and shorts separately - same candle is good for shorts and longs sometimes 
def get_roi_bool(base_row : pd.Series, cur_row: pd.Series, tolerance : float):
    roi = base_row['close']/cur_row['close'] # long 
    if roi-1 >= tolerance:
        return 1 
    
    # shorts 
    roi = base_row['close']/cur_row['close'] # long 
    if 1-roi > tolerance and 1-roi > 0:
        return 0 
    
    return 0.5

df=df.iloc[500:700].reset_index(drop=True)
df['index']=df.index

# algo 
t=0.75/100 # tolerance 
N=len(df)-1 # last index of df   
roi_cat=[0.5 for i in range(N)] # bool roi - 0 short, 0.5 do nothing, 1 long 


def scan_left(base_row:pd.Series, roi_cat : list, df:pd.DataFrame,n : int):    
    while n>0:
        cur_row=df.iloc[n]
        roi=get_roi_bool(base_row=base_row,cur_row=cur_row,tolerance=t)
        if roi  != 0.5: # you dont want to switch a long into nothing the closer you get to it 
            roi_cat[n]=roi
        n=n-1   
    return roi_cat
        
while N>=0:
    base_row=df.iloc[N]
    roi_cat=scan_left(base_row=base_row,roi_cat=roi_cat,df=df,n=N)    
    N=N-1
    

df2=pd.DataFrame(roi_cat,columns=['roi_cat'])
df2.to_csv('winners.csv')

df2['short_flg']=df2['roi_cat'].apply(lambda x: x == 0  ).astype(int)
df2['long_flg']=df2['roi_cat'].apply(lambda x: x == 1  ).astype(int)
df2['nothing_flg']=df2['roi_cat'].apply(lambda x: x==0.5).astype(float)
df2['long_pa']=df2['long_flg']*df['close']
df2['short_pa']=df2['short_flg']*df['close']
axes=mf.plot_candlestick(df=df)
axes[1].plot(df2.index,df2['long_pa'],'^g')
axes[1].plot(df2.index,df2['short_pa'],'vr')
axes[1].plot(df.index,df['close'],'--k')
axes[1].set_ylim(axes[0].get_ylim())

msk=df2['nothing_flg']==1
df2.to_csv('./winners.csv')
plt.show()



