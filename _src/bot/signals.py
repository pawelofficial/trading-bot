

from typing import Any
import pandas as pd
import matplotlib.pyplot as plt 
import os 

class signals:
    def __init__(self, df=pd.DataFrame({})) -> Any:
        self.df=df.copy()
        self.this_dir=os.path.dirname(os.path.abspath(__file__))
        self.data_fp=os.path.join(self.this_dir,'data')
        pass
    
    def signal_wave(self,df,N=10):
        threshold=0.8
        df=df.copy()
        
        df['dif']=(df['close']-df['open']).abs()
        df['flat']=df['dif']<df['dif'].mean()*0.2
        
        df['green']=((df['close']-df['open']>0) | (df['flat']==1) ).astype(int)
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
                     , additional_lines = None # top chart additional line 
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
        ax[0].plot(df[msk].index, df[msk]['low']*df[msk]['LONGS_SIGNAL'],'^g')
    if 'SHORTS_SIGNAL' in df.columns:
        msk=df['SHORTS_SIGNAL']==1
        ax[0].plot(df[msk].index, df[msk]['high']*df[msk]['SHORTS_SIGNAL'],'vr')
        
        
        
    if not shorts_ser.empty:
        mask=shorts_ser>0
        ax[0].plot(shorts_ser.index[mask],shorts_ser[mask],'vr')

    if not longs_ser.empty:
        mask=longs_ser>0
        ax[0].plot(longs_ser.index[mask],longs_ser[mask],'^g')
        
    if not real_longs.empty:
        ax[0].plot(real_longs.index,real_longs,'og')

    if not real_shorts.empty:
        ax[0].plot(real_shorts.index,real_shorts,'or')
        
    if additional_lines is not None:
        for tup in additional_lines:

            which_chart=tup[0]
            series=tup[1]
            ax[which_chart].plot(series.index,series,'-',label=series.name)
        
    ax[0].legend()
    plt.show()
    return ax 
    
if __name__=='__main__':
    df=pd.read_csv('./data/data.csv',sep='|')[:200]
    s,df=signals().signal_wave(df)
    longs=s*df['open']
    plot_candlestick(df,longs_ser=longs,additional_lines=((0,df['ema10']) ,(0,df['ema15'])  )   )
#    s,df=signals().signal_wave(df)
    
    
    