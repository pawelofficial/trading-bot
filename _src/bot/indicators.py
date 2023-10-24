
import pandas as pd 

import ta.momentum as tam 
import ta.volume as tav 
import ta.volatility as tavl
import ta.trend as tat 
import ta.others as tao
import os 

import numpy as np  
import pandas as pd 
import random
import torch 
from scipy.stats import norm
import matplotlib.pyplot as plt 
import datetime 
import logging 
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',filename='indicators.log',filemode='w')



class indicators:
    def __init__(self,fp=None,df=None):
        self.this_dir=os.path.dirname(os.path.abspath(__file__))
        self.data_fp=os.path.join(self.this_dir,'data')

        self.nlq_precision=4
        self.nlq=self.make_nlq(N=50)

        
        self.tformat='%Y-%m-%d %H:%M:%S'
        
        self.funs_d={
            'fun_ema_distance':self.fun_ema_distance,
            'fun_how_green':self.fun_how_green,
            'fun_how_green_streak':self.fun_how_green_streak,
            'fun_cumdist':self.fun_cumdist,
            'fun_rsi':self.fun_rsi,
            'fun_rsi2':self.fun_rsi2,
            'fun_kama':self.fun_kama,
            'fun_ao':self.fun_ao,
            'fun_adi':self.fun_adi,
            'fun_chaikin':self.fun_chaikin,
            'fun_eom':self.fun_eom,
            'fun_atr':self.fun_atr,
            'fun_pband':self.fun_pband,
            'fun_wband':self.fun_wband,
            #'fun_adx':self.fun_adx,
            'fun_macd':self.fun_macd,
            'fun_macd2':self.fun_macd2,
            'fun_aroon':self.fun_aroon,
            'fun_cci':self.fun_cci,
            'fun_cumret':self.fun_cumret,
            'fun_dlr':self.fun_dlr,
            'fun_relative_volatility':self.fun_relative_volatility,
            'fun_donchian_pband':self.fun_donchian_pband,
            'fun_donchian_wband':self.fun_donchian_wband,
            'fun_mass':self.fun_mass
        }
        #self._df=None
        if df is not None:
            self.df=df

        if fp is not None:
            self.read_df(fp)
        
        self.basic_columns=list(self.df.columns)
        self.quantile_columns=[]
        self.fun_columns=[]

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("Expected a pandas DataFrame!")
        self._df = new_df
        #self.basic_columns = new_df.columns  # Update the columns attribute when a new dataframe is set

    
    # returns list of values betweeb 0-1 inclusive with non linear distribution
    def make_nlq(self,N=50, plot_res=False ): 
        x=[i/N for i in range(N+1)]                             # linear percentiles 
        f= lambda x: np.round(1/(1 + np.exp(-(x-0.5 )*10 )),self.nlq_precision) # sigmoid function is the best function out there anon 
        yy=[f(i) for i in x]
        yy=(yy-min(yy))/(max(yy)-min(yy))
        if plot_res:
            import matplotlib.pyplot as plt 
            print(x)
            print(yy)
            plt.plot(x,yy,'-o')
            plt.show()
            exit(1)
        return yy  
    
    def bucketize_col(self, col, window=300, df=None):
        if df is None:
            df = self.df

        new_columns = {}

        q1 = self.nlq[0]  # nlq - list of quantiles, first value is zero
        quantile1 = df[col].rolling(window=window).quantile(quantile=q1)
        for q2 in self.nlq[1:]:
            nq1 = str(q1).replace('0.', '')[:self.nlq_precision + 1]
            nq2 = str(q2).replace('0.', '')[:self.nlq_precision + 1]
            colname = f'{col}_q_{nq1}_{nq2}'
            logging.info(msg=f'{colname} {q1}  {q2}')

            quantile1 = df[col].rolling(window=window).quantile(quantile=q1)
            quantile2 = df[col].rolling(window=window).quantile(quantile=q2)

            new_columns[colname] = ((df[col] >= quantile1) & (df[col] < quantile2)).astype(int)
            self.quantile_columns.append(colname)

            # Prepare for next iteration
            q1 = q2

        self.df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        #return df

    def bucketize_df(self):
        for name,f in self.funs_d.items():
            col_name=f()    
            #print(name,f)
            self.bucketize_col(col=col_name)
            

    def dump_df(self,df=None,fp=None,cols=None,fname='indicators'):
        if cols is None:
            cols=self.basic_columns+self.quantile_columns+self.fun_columns
        if df is None:
            df=self.df
        if fp is None:
            fp=os.path.join(self.data_fp,f'{fname}.csv')
        df[cols].to_csv(fp,sep='|',index=False)
        return df,fp
    
    def read_df(self,fp=None,sep='|'):
        self.df=pd.read_csv(fp,sep=sep)
    
    def aggregate_df(self,df=None,scale=15
                     ,src_col : str = 'timestamp'
                     ,cols : list = ['open','close','low','high','volume']
                     ,timestamp_name='timestamp'
                     ,inplace=True):
    
        if df is None:
            df=self.df.copy()


        floor_dt =  lambda df,str_col, scale: df[str_col].apply(lambda x: x -
                        datetime.timedelta(
                        minutes=(x.minute) % scale,
                        seconds= x.second))
        
        floor_dt =  lambda df,str_col, scale: df[str_col].apply(lambda x: datetime.datetime.strptime(x,self.tformat) -
                        datetime.timedelta(
                        minutes=( datetime.datetime.strptime(x,self.tformat).minute) % scale,
                        seconds= datetime.datetime.strptime(x,self.tformat).second,
                        microseconds= datetime.datetime.strptime(x,self.tformat).microsecond))
        
        agg_funs_d={  'open':lambda ser: ser.iloc[0],
                     'close':lambda ser: ser.iloc[-1],
                     'high':lambda ser: ser.max(),
                     'low': lambda ser: ser.min(),
                     'volume': lambda ser: ser.sum(numeric_only=True)
                    } 
        dt_col = '_'.join(['ts',str(scale) ])
        
        df[dt_col]=floor_dt(df,'timestamp',scale)
        agg_df=pd.DataFrame({})
        agg_df[dt_col]=df[dt_col].unique().copy()
        for col in cols:
            g=df[[col,dt_col]].groupby([dt_col])
            ser=g.apply(agg_funs_d[col])[col].reset_index(name=col)
            agg_df=agg_df.merge(ser,left_on=dt_col,right_on=dt_col)


        agg_df.rename(columns={dt_col:timestamp_name},inplace=True)
        # add epoch column to agg_df based on start_timestamp
        agg_df['epoch']=agg_df[timestamp_name].apply(lambda x: int(datetime.datetime.timestamp(datetime.datetime.strptime(x,self.tformat))))
        
        
        if inplace:
            self.df=agg_df
        
        return agg_df

    # my indicators 
    def fun_ema_distance(self, src_col='close', window=12):
        colname = f'f-emadist-{src_col}-{window}'
        self.fun_columns.append(colname)
        f = lambda df, col, window: (df[col] - df[col].ewm(span=window).mean()) / df[col].rolling(window=window).std() # how many stds away are we from ema
        self.df[colname] = f(df=self.df, col=src_col, window=window)
        return colname
    
    def fun_how_green(self, window=12):
        colname = f'f-hg-{window}'
        self.fun_columns.append(colname)
        f = lambda df, window: (df['close'] > df['open']).rolling(window=window).mean()
        self.df[colname] = f(df=self.df, window=window)
        return colname

    def fun_how_green_streak(self,window=12): # how many consecutive greans 
        colname='f-green-streak'
        self.fun_columns.append(colname)
        green = (self.df['close'] > self.df['open']).astype(int)
        streak = [0] * len(self.df)
        current_streak = 0
        for i in range(len(self.df)):
            if green[i] == 1:
                current_streak += 1
                streak[i] = current_streak
            else:
                current_streak = 0
                streak[i] = 0
        #return streak

        self.df[colname] = streak 
        self.df[colname]=self.df[colname].rolling(window=window).mean()
        return colname
        
    def fun_cumdist(self, src_col='close', window=12):
        colname = f'f-cumdist-{src_col}-{window}'
        self.fun_columns.append(colname)
        f = lambda df, col, window: ((df[col] - df[col].ewm(span=window).mean())).rolling(window=window).sum()
        self.df[colname] = f(df=self.df, window=window, col=src_col)
        return colname

    # momentum indicators
    def fun_rsi(self,window=12):
        colname = f'f-rsi-close-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tam.rsi(close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
    
    def fun_rsi2(self,window=36):
        colname = f'f-rsi2-close-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tam.rsi(close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
    
    
    def fun_kama(self, window=25, pow1=25, pow2=10):
        colname = f'f-kama-close-{window}-{pow1}-{pow2}'
        self.fun_columns.append(colname)
        f = lambda df, window, pow1, pow2: tam.kama(close=df['close'], window=window, pow1=pow1, pow2=pow2)
        self.df[colname] = f(df=self.df, window=window, pow1=pow1, pow2=pow2)
        return colname
    
    def fun_ao(self, window1=5, window2=34):
        colname = f'f-ao-{window1}-{window2}'
        self.fun_columns.append(colname)
        f = lambda df, window1, window2: tam.awesome_oscillator(high=df['high'],
                                                                low=df['low'],
                                                                window1=window1,
                                                                window2=window2)
        self.df[colname] = f(df=self.df, window1=window1, window2=window2)
        return colname
    
    # volume indicators 
    def fun_adi(self ):
        colname = f'f-adi'
        self.fun_columns.append(colname)
        f= lambda df : tav.acc_dist_index(df['high'],df['low'],df['close'],df['volume'])
        self.df[colname]=f(df=self.df)
        return colname
        
    def fun_chaikin(self, window : int = 20):
        colname = f'f-chaikin-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tav.chaikin_money_flow(df['high'],df['low'],df['close'],df['volume'],window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
                
    def fun_eom(self, window : int = 14 ):
        colname = f'f-eom-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tav.ease_of_movement(df['high'],df['low'],df['volume'],window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
        
    # volatility indicators 
    def fun_atr(self, window : int = 14 ):
        colname = f'f-atr-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tavl.average_true_range(df['high'],df['low'],df['close'],window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
    
    def fun_pband(self,  window : int = 20, window_dev : int = 2 ):
        colname = f'f-pband-close-{window}-{window_dev}'
        self.fun_columns.append(colname)
        f= lambda df,window,window_dev : tavl.bollinger_pband(df['close'],window,window_dev)
        self.df[colname]=f(df=self.df,window=window,window_dev=window_dev)
        return colname
        
    def fun_wband(self, window : int = 20, window_dev : int = 2 ):
        colname = f'f-wband-close-{window}-{window_dev}'
        self.fun_columns.append(colname)
        f= lambda df,window,window_dev : tavl.bollinger_wband(df['close'],window,window_dev)
        self.df[colname]=f(df=self.df,window=window,window_dev=window_dev)
        return colname
    
    def fun_donchian_pband(self, window : int = 20 ):
        colname = f'f-pdonchian-close-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tavl.donchian_channel_pband(df['high'],df['low'],df['close'],window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
        
    def fun_donchian_wband(self, window : int = 20 ):
        colname = f'f-wdonchian-close-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tavl.donchian_channel_wband(df['high'],df['low'],df['close'],window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
        
    def fun_adx(self, window : int = 14 ):
        colname = f'f-adx-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tat.adx(high=df['high'],low=df['low'], close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
        
    def fun_macd(self, window_slow : int = 26,window_fast: int = 12 ):
        colname = f'f-macd-{window_slow}-{window_fast}'
        self.fun_columns.append(colname)
        f= lambda df,window_slow,window_fast : tat.macd(close=df['close'],window_slow=window_slow,window_fast=window_fast)
        self.df[colname]=f(df=self.df,window_slow=window_slow,window_fast=window_fast)
        return colname

    def fun_macd2(self, window_slow : int = 12,window_fast: int = 6 ):
        colname = f'f-macd2-{window_slow}-{window_fast}'
        self.fun_columns.append(colname)
        f= lambda df,window_slow,window_fast : tat.macd(close=df['close'],window_slow=window_slow,window_fast=window_fast)
        self.df[colname]=f(df=self.df,window_slow=window_slow,window_fast=window_fast)
        return colname

    
    def fun_mass(self, window_slow : int = 25,window_fast: int = 9 ):
        colname = f'f-mass-{window_slow}-{window_fast}'
        self.fun_columns.append(colname)
        f= lambda df,window_slow,window_fast : tat.macd(close=df['close'],window_slow=window_slow,window_fast=window_fast)
        self.df[colname]=f(df=self.df,window_slow=window_slow,window_fast=window_fast)
        return colname
        
    def fun_aroon(self, window : int = 25 ):
        colname = f'f-aroon-close-{window}'
        self.fun_columns.append(colname)
        f= lambda df,window : tat.aroon_down(close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        return colname
        
    def fun_cci(self, window : int = 20, constant=0.0015 ):
        colname = f'f-cci-{window}-{constant}'
        self.fun_columns.append(colname)
        f= lambda df,window,constant : tat.cci(high=df['high'],low=df['low'],close=df['close'], window=window, constant=constant )
        self.df[colname]=f(df=self.df,window=window,constant=constant)
        return colname
    
    # other indicators 
    def fun_cumret(self ):
        colname = f'f-cumret'
        self.fun_columns.append(colname)
        f= lambda df : tao.cumulative_return(close=df['close'])
        self.df[colname]=f(df=self.df)
        return colname
    
    def fun_dlr(self ):
        colname = f'f-dlr'
        self.fun_columns.append(colname)
        f= lambda df : tao.daily_log_return(close=df['close'])
        self.df[colname]=f(df=self.df)
        return colname
    
    def fun_relative_volatility(self,src_col='close',window1=5,window2=25):
        colname=f'f-volatility-{window1}-{window2}'
        self.fun_columns.append(colname)
        f= lambda df,col,window1,window2 : df[col].rolling(window=window1).std()/df[col].rolling(window=window2).std()
        self.df[colname]=f(df=self.df,col=src_col,window1=window1,window2=window2)
        return colname


    
if __name__=='__main__':
    i=indicators(fp='./data/data.csv')
    agg_df=i.aggregate_df()
    i.bucketize_df()
    
    df=i.df
    print(df.shape)                                                     # 1127 columns 
    print(len(i.quantile_columns))                                      # 1100 columns 
    i.dump_df(cols=i.quantile_columns,fname='quantiles_df')
    
    q_df=pd.read_csv('./data/quantiles_df.csv',sep='|')
    print(q_df.shape)                                                   # 1200 
    
    cols1=list(df.columns)
    cols2=list(q_df.columns)
    
    diff=list(set(cols2)-set(cols1))
    
    b=diff[-1] in i.quantile_columns
    print(b)
    print(diff[-1])
    exit(1)    
    print(i.basic_columns)

    
    print(i.quantile_columns)
    print(i.basic_columns)
    print(i.fun_columns)
    # print lens of columns 
    print(len(i.quantile_columns))   #
    print(len(i.basic_columns))
    print(len(i.fun_columns))
    print(i.df.shape)
    

    exit(1)
    
    
    for name,f in i.funs_d.items():
        col_name=f()    
        print(name,f)
        i.bucketize_col(col=col_name)

    i.dump_df(cols=i.quantile_columns,fname='quantiles_df')
    i.dump_df(cols=i.basic_columns+i.fun_columns,fname='indicators_df')
    
