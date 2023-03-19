import pandas as pd 
import os 
import matplotlib.pyplot as plt 

class Utils:
    def __init__(self) -> None:
        self.data_dir = os.path.abspath('data')
    

    def read_csv(self,fname = 'data',fp = None):
        if fp is not None:
            df=pd.read_csv(fp)
            return df 
        
        fp=os.path.join(self.data_dir,fname.replace('.csv','')+'.csv')
        df=pd.read_csv(fp)
        return df 
    
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
    
    
if __name__=='__main__':
    u=Utils()
    df=u.read_csv('BTC-USD2022-01-01_2022-01-03.csv')
    u.plot_candles(df=df)
    plt.show()