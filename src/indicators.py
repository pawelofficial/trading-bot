
import pandas as pd 

import ta.momentum as tam 
import ta.volume as tav 
import ta.volatility as tavl
import ta.trend as tat 
import ta.others as tao

import matplotlib.pyplot as plt


from utils import Utils
import numpy as np  
import pandas as pd 
import random
import torch 
from scipy.stats import norm
import matplotlib.pyplot as plt 

# this scripty script puts together cool indicators, for now the goal is to have 
# 3 momentum indicators             --> done 
# 3 volume indicators               --> done 
# 3 volatility indicators           --> done 
# 3 trend indicators                --> done 
# 3 other indicators                --> done 

# once this is achieved data is either 
    # bucketized                    --> in progress 
    # normalized                    --> maybe 

# once this is achieved this df will be put to pytorch and vectorbot to find good strategy for specific day/week

# once this is achieved strategies will be connected with each other with pytorch again to have a strategy of strategies

class indicators:
    def __init__(self,df) -> None:
        self.df=df
        self.nlq_steepness=15
        self.nlq_accuracy=5
        self.nlq_number=50
        self.nlq=self.make_nlq() # non lineear quantiles 
        self.raw_data_columns=['epoch','timestamp','low','high']

        self.windows_d={
            'windows_1': [10,10,25, 25, 10, 2, 30, 5, 34, 20, 14, 14, 20, 2, 20, 2, 25, 26, 12, 20, 5, 25],
            'windows_2': [20,20,50, 50, 20, 4, 60, 10, 68, 40, 28, 28, 40, 4, 40, 4, 50, 52, 24, 40, 10, 50]
        }
        
        self.which_windows='windows_1'
        windows=self.windows_d[self.which_windows]
        self.funs_d={
            'cum_dist': (self.fun_cumdist, {"window": windows[0],"src_col":"close"  }),
            'how_green': (self.fun_how_green, {"window": windows[0]}),
            'ema_distance': (self.fun_ema_distance, {"src_col": 'close', "window": windows[0]}),
            'rsi':  (self.fun_rsi, {"window": windows[1]}),
            'kama':  (self.fun_kama, {"window": windows[2], "pow1": windows[3], "pow2": windows[4]}),
            'ao': (self.fun_ao, {"window1": windows[5], "window2": windows[6]}),
            'adi': (self.fun_adi, {}),
            'chaikin': (self.fun_chaikin, {"window": windows[7]}),
            'eom': (self.fun_eom, {"window": windows[8]}),
            'atr': (self.fun_atr, {"window": windows[9]}),
            'pband': (self.fun_pband, {"window": windows[10], "window_dev": windows[11]}),
            'wband': (self.fun_wband, {"window": windows[12], "window_dev": windows[13]}),
            'aroon': (self.fun_aroon, {"window": windows[14]}),
            'macd': (self.fun_macd, {"window_slow": windows[15], "window_fast": windows[16]}),
            'cci': (self.fun_cci, {"window": windows[17]}),
            'cumret': (self.fun_cumret, {}),
            'dlr': (self.fun_dlr, {}),
            'relative_volatility': (self.fun_relative_volatility, {"window1": windows[18],"window2":windows[19]  })
        }
        self.q_cols  = lambda : [c for c in list(self.df.columns) if  c[:2] == 'f-'] # columns for quantile calcls start with f- 
        self.b_cols = lambda : [c for c in list(self.df.columns) if  c[:2] == 'q.'] # columns to dump with quantile booleans
        self.trade_pairs=[]
        # true / false positive/negatives
        self.metrics_d={
            'tp':  lambda t1,t2 : torch.mul(torch.where(t1==1,1,0), torch.where(t2==1,1,0)).sum() / len ( torch.where(t1==1)[0])
            ,'tn':  lambda t1,t2: torch.mul( torch.where(t1==0,1,0), torch.where(t2==0,1,0)).sum() / len ( torch.where(t1==0)[0])
            ,'fp': lambda t1,t2 : torch.mul(torch.where(t1==1,1,0)  , torch.where(t2==0,1,0)).sum() / len(torch.where(t1==1)[0]) 
            ,'fn': lambda t1,t2 : torch.mul(torch.where(t1==0,1,0)  , torch.where(t2==1,1,0)).sum() / len(torch.where(t1==0)[0]) 
            }
        
    def refresh_funs_d(self):
        windows=self.windows_d[self.which_windows]
        self.funs_d={
            'cum_dist': (self.fun_cumdist, {"window": windows[0],"src_col":"close"  }),
            'how_green': (self.fun_how_green, {"window": windows[0]}),
            'ema_distance': (self.fun_ema_distance, {"src_col": 'close', "window": windows[0]}),
            'rsi':  (self.fun_rsi, {"window": windows[1]}),
            'kama':  (self.fun_kama, {"window": windows[2], "pow1": windows[3], "pow2": windows[4]}),
            'ao': (self.fun_ao, {"window1": windows[5], "window2": windows[6]}),
            'adi': (self.fun_adi, {}),
            'chaikin': (self.fun_chaikin, {"window": windows[7]}),
            'eom': (self.fun_eom, {"window": windows[8]}),
            'atr': (self.fun_atr, {"window": windows[9]}),
            'pband': (self.fun_pband, {"window": windows[10], "window_dev": windows[11]}),
            'wband': (self.fun_wband, {"window": windows[12], "window_dev": windows[13]}),
            'aroon': (self.fun_aroon, {"window": windows[14]}),
            'macd': (self.fun_macd, {"window_slow": windows[15], "window_fast": windows[16]}),
            'cci': (self.fun_cci, {"window": windows[17]}),
            'cumret': (self.fun_cumret, {}),
            'dlr': (self.fun_dlr, {}),
            'relative_volatility': (self.fun_relative_volatility, {"window1": windows[18],"window2":windows[19]  })
        }
        
        
    # plots stuff and shows it ! 
    def plot_stuff(self,df = None,extra_col : str =None,plot_flag=True):
        if df is None:
            df=self.df 
        import matplotlib.pyplot as plt 
        fig,ax=plt.subplots(2,1,sharex=True)
        
        c=df[extra_col]/(np.max(df[extra_col])-np.min(df[extra_col]))
        ax[0].scatter(df.index,df.close,c=df[extra_col],cmap='Blues',marker='.')
        if 'entry' in df.columns:
            entry_mask=df['entry']==1
            ax[0].plot(df[entry_mask].index,df[entry_mask]['close'] ,'^m')
            ax[0].grid()
        if 'exit' in df.columns:
            exit_mask=df['exit']==1
            ax[0].plot(df[exit_mask].index,df[exit_mask]['close'] ,'vr')
            ax[0].grid()    
        if extra_col is not None:
            ax[1].plot(df.index,df[extra_col])
            ax[1].grid()
        
        if not plot_flag:
            return ax
        plt.show()
        
    def make_nlq(self, plot_res=False ):  # returns list of values betweeb 0-1 inclusive with non linear distribution 
        x=[i/self.nlq_number for i in range(self.nlq_number+1)] # linear percentiles

        f = lambda x: np.exp(3*x)   # your distribution function 
        f = lambda x: np.sin(2*x)
        f= lambda x: np.round(1/(1 + np.exp(-(x-0.5 )*self.nlq_steepness )),self.nlq_accuracy) # sigmoid function is the best function out there anon 

        yy=[f(i) for i in x]
        yy=(yy-min(yy))/(max(yy)-min(yy))
        if len(yy)!=len(list(set(yy))):
            print('incorrect nlq inputs - quantiles overlap due to accuracy')
            raise 

        if plot_res:
            import matplotlib.pyplot as plt 
            print(x)
            print(yy)
            plt.plot(x,yy,'-o')
            plt.show()
            exit(1)
        return yy # gotta figure out sth cool 

    #calculates profit based on trade pairs 
    def calculate_profit(self,trade_size=100):
        max_profit=0
        for d in self.trade_pairs:
            profit=d['pnl']*trade_size - trade_size
            max_profit+=profit 
        return max_profit

    # max profit you could make if you perfectly caught every candle on open-close 
    def calculate_max_profit(self,df=None,long_only=True,trade_size=100):
        if df is None:
            df=self.df
        if long_only:
            msk=df['close']>df['open']
            return np.sum(df[msk]['close']/df[msk]['open']*trade_size - trade_size )
        else:
            return np.sum(np.abs(df['close']-df['open']))
            
    # makes exit column with exit signals based on a function , populate trade pairs 
    def calculate_exits(self,df=None,entry_colname='entry',exit_colname='exit', fun = None,all_rows=False ) -> int: # returns index for an exit  
        if fun is None:
            fun=self.calculate_trtp
        if df is None:
            df=self.df 
        df[exit_colname]=0
        for index,row in df.iterrows(): 
            entry_signal=row[entry_colname] 
            if entry_signal==1 or all_rows:
                exit_index=fun(df=self.df,index=index)
                if exit_index is not None: # this will leave orphans 
                    df.loc[exit_index,'exit']=1
                    d={'entry_index':index,'exit_index':exit_index,'pnl': round(df.loc[exit_index,'close']/df.loc[index,'close'],4) }
                    self.trade_pairs.append(d)

    def calculate_roi(self,df=None,entry_colname='entry',fun=None,inplace=True):
        if fun is None:
            fun=self.calculate_trtp
        if df is None:
            df=self.df 
            
        df['roi']=1
        for index,row in df.iterrows():
            exit_index=fun(df=self.df,index=index)
            if exit_index is None:
                continue 
            df.loc[index,'roi']=df.loc[exit_index,'close']/df.loc[index,'close']
        if not inplace:
            ser=df['roi']
            df.drop(labels='roi',axis=1,inplace=True)
            return ser**3
        

        if False: # normalize ?? 
            df['roi']=df['roi']/(max(df['roi'])-min(df['roi']) )
                    
    # makes entry column with random signal
    def make_random_entries(self,df=None,entry_colname='entry',N=20) -> pd.Series: 
        if df is None:
            df=self.df
        p=N/len(df) # probability of event to yield N events 
        df[entry_colname]=df.apply( lambda x: int(random.randint(0,100)<p*100),axis=1)

    def calculate_tp(self,df,index=100,tp=1.01) ->int :
        row=df.iloc[index]
        cur_price=row['close']
        while index < len(df)-1:
            index +=1 
            row=df.iloc[index]
            price = row['close']
            pnl = price / cur_price 
            if pnl >= tp:
                return index

    def calculate_sl(self,df,index=100,sl=0.99) -> int :
        row=df.iloc[index]
        cur_price=row['close']
        while index < len(df)-1:
            index +=1 
            row=df.iloc[index]
            price = row['close']
            pnl = price / cur_price 
            if pnl < sl:
                return index  
        
    def calculate_trtp(self,df,index=100,trtp=0.99) -> int :
        cur_price=df.loc[index,'close']
        max_pnl=1
        min_pnl=1
        min_index=index
        
        while index < len(df)-1:
            index +=1 
            pnl = df.loc[index,'close'] / cur_price 
            if pnl >= max_pnl:
                max_pnl = pnl 
                cur_price = df.loc[index,'close'] 
            if pnl<=min_pnl: # if trtp didnt happen then return min pnl 
                min_pnl=pnl
                min_index=index
            elif pnl < trtp:
                return index
            
        return min_index
                         
    def bucketizeme(self,df,col,window= 300,inplace = True ):
        # bucketizes column into N buckets based on its history 
        q1=self.nlq[0]
        quantile1=df[col].rolling(window=window).quantile(quantile=q1)
        for i in range(len(self.nlq)-1):                      
            q2=self.nlq[i+1]
            nq1=str(q1).replace('0.','.')
            nq2=str(q2).replace('0.','.')
            colname=f'q{nq1}-{nq2}-{col}'
            quantile2=df[col].rolling(window=window).quantile(quantile=q2)
            df[colname]=((df[col]>=quantile1) & (df[col]<quantile2)).astype(int)
            q1=q2
            quantile1=quantile2
        if inplace:
            self.df=df.copy(deep=True)
            return self.df 
        return df.copy(deep=True)
    # dummy 
    def fun_dummy(self, window : int = 0, src_col : int = 0, inplace = True ):
        colname = f'dummy-{src_col}-{window}'
        f= lambda df,col,window : df[col] - window 
        if inplace:
            self.df[colname]=f(df=self.df,col=src_col,window=window)
            return 
        return f(df=self.df,col=src_col,window=window)
    #  ema 
    def fun_ema_distance(self,src_col='close', window : int = 25):
        colname=f'f-emadist-{src_col}-{window}'
        f = lambda df,col,window : (df[col]-df[col].ewm(span=window).mean())/df[col].rolling(window=window).std() # not sure why i did it this way 
        self.df[colname]=f(df=self.df,col=src_col,window=window)
        
    def fun_how_green(self, window : int = 10):
        colname=f'f-hg-{window}'
        f=lambda df,window :  (df['close']>df['open']).rolling(window=window).mean()
        self.df[colname]=f(df=self.df,window=window)
    
    def fun_cumdist(self,src_col='close', window : int = 10):
        colname=f'f-cumdist-{src_col}-{window}'
        f = lambda df, col, window: ((df[col] - df[col].ewm(span=window).mean()) ).rolling(window=window).sum()
        self.df[colname]=f(df=self.df,window=window,col=src_col)
        
# momentum indicators 
    #  rsi 
    def fun_rsi(self, window: int = 25 , inplace = True ):
        colname = f'f-rsi-close-{window}'
        f= lambda df,window : tam.rsi(close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        
    # kama ema 
    def fun_kama(self, window :int =  10, pow1 : int = 2, pow2 : int = 30,  inplace = True ):

        colname = f'f-kama-close-{window}-{pow1}-{pow2}'
        f= lambda df,window,pow1,pow2: tam.kama(close=df['close'], window=window,pow1=pow1,pow2=pow2) 
        self.df[colname]=f(df=self.df,window=window,pow1=pow1,pow2=pow2)
    # awesome indicator 
    
    def fun_ao(self,window1:int =5 , window2=34, inplace = True ):
        colname=f'f-ao-{window1}-{window2}'
        f= lambda df,window1,window2 : tam.awesome_oscillator(high=df['high'],
                                                              low=df['low'],
                                                              window1=window1,
                                                              window2=window2  )
        self.df[colname]=f(df=self.df,window1=window1,window2=window2)

# volume indicators 
    # adi 
    def fun_adi(self, inplace = True ):
        colname = f'f-adi'
        f= lambda df : tav.acc_dist_index(df['high'],df['low'],df['close'],df['volume'])
        self.df[colname]=f(df=self.df)
    # chaikin 
    def fun_chaikin(self, window : int = 20, inplace = True ):
        colname = f'f-chaikin-{window}'
        f= lambda df,window : tav.chaikin_money_flow(df['high'],df['low'],df['close'],df['volume'],window)
        self.df[colname]=f(df=self.df,window=window)
        
    # ease of movement 
    def fun_eom(self, window : int = 14, inplace = True ):
        colname = f'f-eom-{window}'
        f= lambda df,window : tav.ease_of_movement(df['high'],df['low'],df['volume'],window)
        self.df[colname]=f(df=self.df,window=window)

# volatility indicators 
    # average_true_range
    def fun_atr(self, window : int = 14, inplace = True ):
        colname = f'f-atr-{window}'
        f= lambda df,window : tavl.average_true_range(df['high'],df['low'],df['close'],window)
        self.df[colname]=f(df=self.df,window=window)
        
    # boiilinger percentage indicator - use 
    def fun_pband(self,  window : int = 20, window_dev : int = 2, inplace = True ):
        colname = f'f-pband-close-{window}-{window_dev}'
        f= lambda df,window,window_dev : tavl.bollinger_pband(df['close'],window,window_dev)
        self.df[colname]=f(df=self.df,window=window,window_dev=window_dev)
        
    # boilinger band width - use 
    def fun_wband(self, window : int = 20, window_dev : int = 2, inplace = True ):
        colname = f'f-wband-close-{window}-{window_dev}'
        f= lambda df,window,window_dev : tavl.bollinger_wband(df['close'],window,window_dev)
        self.df[colname]=f(df=self.df,window=window,window_dev=window_dev)

# trend indicators 
    # adx  - wywala sie 
    def fun_adx(self, window : int = 14, inplace = True ):
        colname = f'f-adx-{window}'
        f= lambda df,window : tat.adx(high=df['high'],low=df['low'], close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        
    # macd 
    def fun_macd(self, window_slow : int = 26,window_fast: int = 12,  inplace = True ):
        colname = f'f-macd-{window_slow}-{window_fast}'
        f= lambda df,window_slow,window_fast : tat.macd(close=df['close'],window_slow=window_slow,window_fast=window_fast)
        self.df[colname]=f(df=self.df,window_slow=window_slow,window_fast=window_fast)
        
    # aroon 
    def fun_aroon(self, window : int = 25,  inplace = True ):
        colname = f'f-aroon-close-{window}'
        f= lambda df,window : tat.aroon_down(close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        
    # dpo - wywala sie 
    def fun_dpo(self, window : int = 20, inplace = True ):
        colname = f'f-dpo-close-{window}'
        f= lambda df,window : tat.dpo(close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        
    # cci
    def fun_cci(self, window : int = 20, constant=0.0015, inplace = True ):
        colname = f'f-cci-{window}-{constant}'
        f= lambda df,window,constant : tat.cci(high=df['high'],low=df['low'],close=df['close'], window=window, constant=constant )
        self.df[colname]=f(df=self.df,window=window,constant=constant)
    
# other indicators 
    # cumulative return
    def fun_cumret(self, inplace = True ):
        colname = f'f-cumret'
        f= lambda df : tao.cumulative_return(close=df['close'])
        self.df[colname]=f(df=self.df)
        
    # dlr
    def fun_dlr(self,inplace = True ):
        colname = f'f-dlr'
        f= lambda df : tao.daily_log_return(close=df['close'])
        self.df[colname]=f(df=self.df)
        
    # relative volatility 
    def fun_relative_volatility(self,src_col='close',window1=5,window2=25,inplace=True):
        colname=f'f-volatility'
        f= lambda df,col,window1,window2 : df[col].rolling(window=window1).std()/df[col].rolling(window=window2).std()
        self.df[colname]=f(df=self.df,col=src_col,window1=window1,window2=window2)

    # returns series with entry signal - big green 
    def signal_big_green(self,df):
        df=df.copy()
        next_close=df['close'].shift(-1)
        next_open=df['open'].shift(-1)
        candle_size = (df['close']-df['open']).abs()
        mean_candle_size = candle_size.mean()
        std_candle_size = candle_size.std()
        green_mask=next_close>next_open
        n=1
        big_mask=(next_close-next_open)>mean_candle_size+n*std_candle_size
        mask=green_mask&big_mask
        return mask.astype(int)

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
                while df.loc[j-1,'green']==1 and j>0 :                      # if previous candle is green make it green
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

            
        return df 



# plts candlestick fren 
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
    u=Utils()
    df=u.read_csv('./src/data/raw/data.csv')
    df=df.iloc[:100]
    i=indicators(df)
    df=i.signal_wave(df=df)
    msk=df['wave_signal']==1
    longs_ser=df['low'][msk]

    
    additional_lines=[(0,df['ema10']),(0,df['ema15']) ]
    plot_candlestick(df,longs_ser=longs_ser,  additional_lines=additional_lines)

#    exit(1)
#    signal=i.signal_big_green(df)
#    msk=signal>0
#    longs_ser=df['close'][msk]