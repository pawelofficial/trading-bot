

from typing import Any
import pandas as pd
import matplotlib.pyplot as plt 
import os 
from rich.traceback import install
import random 
import logging 
import numpy as np 
install()

# Set up logging
def setup_logging():
    logging.basicConfig(filename='./logs/signals.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.warning('starting ')


class signals:
    def __init__(self, df=pd.DataFrame({})) -> Any:
        self.df=df.copy()
        self.this_dir=os.path.dirname(os.path.abspath(__file__))
        self.data_fp=os.path.join(self.this_dir,'data')
        pass
    

    # new version of wave signal 
    def signal_wave2(self,df,ema1=5,ema2=10,shift_no=1):
        threshold=1
        N=4
        flat_threshold=0.2
        df=df.copy()
        df['wave_signal']=0 # declare wave signal 
        
        df['dif'] = df['close'] - df['open']                             # Difference between open and close
        df['flat'] = df['dif'].abs() < df['dif'].abs().mean() * flat_threshold
        df['green'] = (df['dif'] > 0) | (df['flat'] == True)
        
        df[f'ema{ema1}']=df['close'].ewm(span=ema1).mean()
        df[f'ema{ema2}']=df['close'].ewm(span=ema2).mean()
    
        # first condition -> ema10 > ema15
        df['condition1']=(df[f'ema{ema1}']>df[f'ema{ema2}']).astype(int)  
        
        # second condition -> mid of candle above ema1 
        df['condition2']=(df['close']+df['open'])/2  -df[f'ema{ema1}'] >0
        
        # remove condition2 orphans 
        df['tmp_wave']=df['condition1'] & df['condition2']
        df['tmp_orphans']=0
        for no in range(1,len(df)-1):
            next_row=df.iloc[no+1]['tmp_wave']
            previous_row=df.iloc[no-1]['tmp_wave']
            if next_row==0:
                if previous_row==0:
                    df.loc[no,'tmp_orphans']=1
        df['tmp_wave']=df['tmp_wave'] & ~df['tmp_orphans']


        # third condition -> green candles prior to ema signal 
        df['condition3']=0
        for no,row in df.iterrows():
            if row['tmp_wave']==1:  # if current row is a wave 
                j=no            
                while j>0 and df.loc[j-1,'green']==1: # if previous candle is green make it green
                    df.loc[j-1,'condition3']=1    
                    j=j-1
        df['tmp_wave']=df['tmp_wave']  | df['condition3']


        # fourth condition -> trailing red candles 
        df['condition4']=0
        for no,row in df.iterrows():
            next_rows=df.iloc[no+1:no+N]['tmp_wave'].tolist()
            if next_rows.count(1)==0:
                df.loc[no,'condition4']=1
        df['condition4']=df['condition4'] & ~df['green']
        df['tmp_wave']=df['tmp_wave']  & ~df['condition4']
        df['wave_signal'] = df['tmp_wave']
        
        # fifth condition - catch green or flat candles prior to wave 
        
        if shift_no is not None:
            df['wave_signal']=df['wave_signal'].shift(shift_no).fillna(0).astype(float)
        
        return df['wave_signal'],df
    
    def signal_wave(self,df,N=5):
        threshold=0.8
        df=df.copy()
        
        df['dif']=(df['close']-df['open']).abs()
        df['flat']=df['dif']<df['dif'].mean()*0.2
        
        df['green']=((df['close']-df['open']>0) & (df['flat']==0) ).astype(int)
        ema10=df['close'].ewm(span=10).mean()
        ema15=df['close'].ewm(span=15).mean()
        df['ema10']=ema10
        df['ema15']=ema15
        df['ema_signal']=(ema10>ema15).astype(int)
        df['wave_signal']=0
        
        for no,row in df.iterrows(): 
            next_rows=df.iloc[no+1:no+N]['ema_signal'].tolist()
            if next_rows.count(1)>=int(threshold*N):
                df.loc[no,'wave_signal']=1
                
        # add additional green candles prior to wave 
        for no, row in df.iterrows():
            if row['wave_signal']==1:                                       # if this candle is a wave
                j=no
                while j > 0 and df.loc[j-1,'green']==1 and j>0 :            # if previous candle is green make it green
                    df.loc[j-1,'wave_signal']=1                                 
                    j=j-1
                    
        # add additional green calndles after a wave
        for no, row in df.iterrows():
            if row['wave_signal']==1:                                       # if this candle is a wave
                j=no
                while df.loc[j+1,'green']==1 and j<df.index.max() :         # if next candle is green make it green
                    df.loc[j+1,'wave_signal']=1
                    j=j+1
            
        # remove last candles if they are red or flat 
        for no, row in df.iterrows():
            if no < df.index.max() and df.loc[no+1,'wave_signal']==0  :             # if next candle is not a wave
                j = no
                while j > 0 and (df.loc[j,'green']==0 or df.loc[j,'flat']==1) :     # if current candle is red or flat make it red 
                    df.loc[j,'wave_signal'] = 0
                    j = j-1

        df['wave_signal_start']=0 # first three rows of wave 

        for no,row in df.iterrows():
            if no==0:
                continue 
            prev_row=df.iloc[no-1]['wave_signal']
            next_five_rows=df.iloc[no+1:no+5]['wave_signal'].tolist()
            if prev_row==0 and all(next_five_rows)==1:
                df.loc[no+1:no+2,'wave_signal_start']=1
                
        df['wave_signal_end']=0
        for no, row in df.iterrows():
            if no==0:
                continue
            prev_row=df.iloc[no-1]['wave_signal']
            this_row=df.iloc[no]['wave_signal']
            next_five_rows=df.iloc[no+1:no+2]['wave_signal'].tolist()
            
            if this_row==0 and prev_row==1 and all(next_five_rows)==0:
                df.loc[no-3:no-1,'wave_signal_end']=1
                #print(no)
            

        return df['wave_signal'],df
    
    def dump_df(self,df=None,fp=None,cols=None,fname='indicators'):
        if df is None:
            df=self.df
        if cols is None:
            cols=df.columns
        if fp is None:
            fp=os.path.join(self.data_fp,f'{fname}.csv')
        df[cols].to_csv(fp,sep='|',index=False)
        return df,fp


    def backtest_random(self,df=None, money=10000,  price_col=['open', 'close']):
        if df is None:
            df = self.df
        df=df.copy()
        df['entry']=random.choices([0,1],k=len(df))
        df['exit']=random.choices([0,1],k=len(df))
        money,r=self.backtest(df=df,money=money,entry_signal='entry',exit_signal='exit',price_col=price_col)
        return money, r 


    # simple backtest 
    def backtest(self, df=None, money=10000, entry_signal='entry', exit_signal='exit', price_col=['open', 'close']):
        money_zero=money
        if df is None:
            df = self.df
        df=df.copy()
        amo = 0
        for _, row in df.iterrows():
            #print(amo,money,row[price_col].to_dict(), row[entry_signal], row[exit_signal] )
            #input('wait')
            d = row.to_dict()
            if d[entry_signal] == 1 and money >=0 :  # Check for non-zero price
                logging.warning('buying')
                logging.warning(f"amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")
                amo += money / d[price_col[0]]
                money = 0
                logging.warning(f"amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")
            elif d[exit_signal] == 1 and amo>=0 :
                logging.warning('selling')
                logging.warning(f"amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")
                money += amo * d[price_col[1]]
                amo = 0
                logging.warning('sold')
                logging.warning(f"amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")



        if amo != 0:  # Only add to money if there are holdings left
            money += amo * df.iloc[-1][price_col[1]]
            
        return money, money/money_zero*100

        
                


def plot_df(df, top_chart_cols=['col1', 'col2'], bottom_chart_cols=['col1', 'col3']):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    # Top chart
    for col in top_chart_cols:
        ax[0].scatter(df.index, df[col], label=col)
    ax[0].legend()
    ax[0].set_title('Top Chart: ' + ', '.join(top_chart_cols))
    
    
    # Bottom chart
    for col in bottom_chart_cols:
        ax[1].scatter(df.index, df[col], label=col)
    ax[1].legend()
    ax[1].set_title('Bottom Chart: ' + ', '.join(bottom_chart_cols))
    
    plt.tight_layout()
    ax[0].grid(True,which="both",ls="-", color='0.65')
    ax[1].grid(True,which="both",ls="-", color='0.65')
    plt.show()

    
def plot_candlestick(df
                     , shorts_ser:pd.Series = pd.Series(dtype=float)  # my shorts 
                     , longs_ser:pd.Series = pd.Series(dtype=float)   # my longs 
                     , real_longs:pd.Series = pd.Series(dtype=float)  # model longs 
                     , real_shorts:pd.Series = pd.Series(dtype=float) # model shorts 
                     , purple_ser:pd.Series = pd.Series(dtype=float) # model shorts 
                     , additional_lines = None # top chart additional line 
                     , extra_sers=[]
                     ):
    plt.rcParams['axes.facecolor'] = 'y'
    low=df['low']
    high=df['high']
    open=df['open']
    close=df['close']
    # mask for candles 
    green_mask=df['close']>=df['open']
    red_mask=df['open']>df['close']
    up=df[green_mask]
    down=df[red_mask]
    # colors
    col1='green'
    black='black'
    col2='red'

    width = .4
    width2 = .05

    fig,ax=plt.subplots(2,1)
    ax[0].bar(up.index,up['high']-up['close'],width2,bottom=up['close'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['low']-up['open'],width2, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['close']-up['open'],width, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(down.index,down['high']- down['close'],width2,bottom=down['close'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['low']-  down['open'],width2,bottom=down['open'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['close']-down['open'],width,bottom=down['open'],color=col2,edgecolor=black)
        
    if 'LONGS_SIGNAL' in df.columns:
        msk=df['LONGS_SIGNAL']==1
        ax[0].plot(df[msk].index, df[msk]['low']*df[msk]['LONGS_SIGNAL'], '^', color='lightgreen')
    if 'SHORTS_SIGNAL' in df.columns:
        msk=df['SHORTS_SIGNAL']==1
        ax[0].plot(df[msk].index, df[msk]['high']*df[msk]['SHORTS_SIGNAL'],'vr')
        
        
        
    if not shorts_ser.empty:
        mask=shorts_ser>0
        ax[0].plot(shorts_ser.index[mask],shorts_ser[mask],'vr')

    if not longs_ser.empty:
        mask=longs_ser>0
        ax[0].plot(longs_ser.index[mask],longs_ser[mask], '^', color='lightgreen')
        
    if not real_longs.empty:
        ax[0].plot(real_longs.index,real_longs,'og')

    if not real_shorts.empty:
        ax[0].plot(real_shorts.index,real_shorts,'or')
        
    if not purple_ser.empty:
        mask=purple_ser>0
        ax[0].plot(purple_ser.index[mask],purple_ser[mask],'om')
        
    for tup in extra_sers:
        which_chart=tup[0]
        marker=tup[1]
        ser=tup[2]
        mask=ser>0
        ax[which_chart].plot(ser.index[mask],ser[mask],marker)
        
        
    if additional_lines is not None:
        for tup in additional_lines:
            which_chart=tup[0]
            series=tup[1]
            ax[which_chart].plot(series.index,series,'-',label=series.name)
        
    ax[0].legend()
    plt.show()
    return ax 
    
    
if __name__=='__main__':
    setup_logging()
    df=pd.read_csv('./data/signals_df.csv',sep='|').reset_index()
    df['close']=df['close'] / df['close'].ewm(span=100, adjust=False).mean()
    df['open']=df['open'] / df['open'].ewm(span=100, adjust=False).mean()
    
    df['wave_signal']=df['wave_signal'].astype(int)
    df['exit_signal']=abs(df['wave_signal']-1).astype(int)
    #print(df[['wave_signal','exit_signal']].head(25))
    
    money,r=signals().backtest_random(df)
    print(money,r)
    money,r=signals().backtest(df,entry_signal='wave_signal',exit_signal='exit_signal' )
    print(money,r)
    exit(1)
        
    
if __name__=='__main__':
    df=pd.read_csv('./data/signals_df.csv',sep='|')[300:400].reset_index()
    s,df=signals().signal_wave2(df)
    #n1,n2=300,700
    #df=df[n1:n2]#.reset_index()

#    print(df['wave_signal'].head(25) )

    #s,df=signals().signal_wave(df)
    longs=s*df['open']
    condition3=df['condition3']*df['high']
    
    extra_sers=[(0,'xr',condition3) ]

    plot_candlestick(df,longs_ser=longs,extra_sers=extra_sers
                     ,  additional_lines=((0,df['ema5']) ,(0,df['ema10'])  )   )
#    s,df=signals().signal_wave(df)
    
    
    