import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import datetime 

class Utils:
    def __init__(self) -> None:
        self.data_dir = os.path.abspath('data')
    
    # read csv and cast timestamp column to a timestamp 
    def read_csv(self,fname = 'data',fp = None,cast_ts=True,ts_col='timestamp',tformat='%Y-%m-%d %H:%M:%S'):
        if fp is not None:
            df=pd.read_csv(fp)
            return df 
        
        fp=os.path.join(self.data_dir,fname.replace('.csv','')+'.csv')
        df=pd.read_csv(fp)
        
        if cast_ts:
            to_ts = lambda x,tformat: datetime.datetime.strptime(x,tformat)
            df[ts_col]=df[ts_col].apply(lambda x: to_ts(x,tformat))
            
        return df
     
    def dump_df(self,df,fname='data'):
        fname=fname.replace('.csv','')+'.csv'
        fp=os.path.join(self.data_dir,fname)
        df.to_csv(path_or_buf=fp,header=0,index=False)
    
    
    def plot_candles(self,df,
                     sers=[],                   # addition al series to plot on ax1 
                     longs_bl :pd.Series =None, # boolean column to plot on ax 0  with price action 
                     shorts_bl :pd.Series=None,show=True):
        def fun(ax): # plots bars 
            ax.bar(up.index,up['high']-up['close'],width2,bottom=up['close'],color=col1,edgecolor=black)
            ax.bar(up.index,up['low']-up['open'],width2, bottom=up['open'],color=col1,edgecolor=black)
            ax.bar(up.index,up['close']-up['open'],width, bottom=up['open'],color=col1,edgecolor=black)
            ax.bar(down.index,down['high']- down['close'],width2,bottom=down['close'],color=col2,edgecolor=black)
            ax.bar(down.index,down['low']-  down['open'],width2,bottom=down['open'],color=col2,edgecolor=black)
            ax.bar(down.index,down['close']-down['open'],width,bottom=down['open'],color=col2,edgecolor=black)
            return ax 
        
        col1='green'
        black='black'
        col2='red'
        # mask for candles 
        green_mask=df['close']>=df['open']
        red_mask=df['open']>df['close']
        up=df[green_mask]
        down=df[red_mask]

        width = .4
        width2 = .05
        fig,ax=plt.subplots(2,1)
        

        if longs_bl is not None:
            msk=longs_bl==True
            tmp_df=df[msk]
            ax[0].plot(tmp_df['index'],tmp_df['close'],'^g',label='longs')
        if shorts_bl is not None:
            msk=shorts_bl==True
            tmp_df=df[msk]
            ax[0].plot(tmp_df['index'],tmp_df['close'],'vr',label='shorts')

        fun(ax[0])
        ax[0].legend(loc='upper left')
        if sers is not None:
            for s in sers: 
                ax[1].plot(s.index,s,'x')

            
        if show:
            plt.show()
    
    # aggregates df to intervals 
    def aggregate(self,df :pd.DataFrame = pd.DataFrame({}), 
                  scale : int = 5, 
                  src_col : str = 'timestamp',
                  cols : list = ['open','close','low','high','volume'],
                  timestamp_name='timestamp'):
        tformat='%Y-%m-%dT%H:%M:%S.%fZ'

    #    floor_dt =  lambda df,str_col, scale: df[str_col].apply(lambda x: datetime.datetime.strptime(x,tformat) -
    #                    datetime.timedelta(
    #                    minutes=( datetime.datetime.strptime(x,tformat).minute) % scale,
    #                    seconds= datetime.datetime.strptime(x,tformat).second,
    #                    microseconds= datetime.datetime.strptime(x,tformat).microsecond))
    #    
        floor_dt =  lambda df,str_col, scale: df[str_col].apply(lambda x: x -
                        datetime.timedelta(
                        minutes=( x.minute) % scale,
                        seconds= x.second))

        agg_funs_d={  'open':lambda ser: ser.iloc[0],
                     'close':lambda ser: ser.iloc[-1],
                     'high':lambda ser: ser.max(),
                     'low': lambda ser: ser.min(),
                     'volume': lambda ser: ser.sum(numeric_only=True)
                    }  

        if src_col not in df.columns:
            print('src col not in df columns')
            raise 
        dt_col = '-'.join(['ts',str(scale) ])
        agg_df=pd.DataFrame({})

        df[dt_col]=floor_dt(df,'timestamp',scale)
    #    df[dt_col]=calculate_fun(df=df,fun_name='floor_dt',str_col='timestamp',scale=scale) # need to write floor column to df  to later groupby it 
        agg_df[dt_col]=df[dt_col].unique().copy()
        for col in cols:
            g=df[[col,dt_col]].groupby([dt_col])
            ser=g.apply(agg_funs_d[col])[col].reset_index(name=col)
            agg_df=agg_df.merge(ser,left_on=dt_col,right_on=dt_col)


        agg_df.rename(columns={dt_col:timestamp_name},inplace=True)
        return agg_df
    
if __name__=='__main__':
    u=Utils()
    df=u.read_csv('BTC-USD2022-01-01_2022-01-03.csv')
    df_agg=u.aggregate(df=df)
    print(df_agg)
    print(df.head(25))
#    u.plot_candles(df=df)
#    plt.show()