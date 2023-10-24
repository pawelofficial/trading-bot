

from typing import Any
import pandas as pd
import matplotlib.pyplot as plt 
import os 
from rich.traceback import install
import random 
import logging 
import numpy as np 
from backtesting import Backtest, Strategy
import datetime
install()
random.seed(1)

# Set up logging
def setup_logging():
    logging.basicConfig(filename='./logs/signals.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',)
    logging.warning('starting ')
    logging.propagate = False

setup_logging()
class signals:
    def __init__(self, df=pd.DataFrame({})) -> Any:
        self.df=df.copy()
        self.this_dir=os.path.dirname(os.path.abspath(__file__))
        self.data_fp=os.path.join(self.this_dir,'data')
        self.normfun=lambda x: x.ewm(span=100).mean()
        self.buy_flg=1   # set to 1 to test switcharoo stradegy  
        self.sell_flg=0
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
            df['wave_signal']=df['wave_signal'].shift(shift_no).fillna(0).astype(int)
        
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

    # backtests baseline strategies - random, hodl, avg
    def backtest_baseline(self,df=None, money=10000,  price_col=['open', 'close'],normalize=True):
        if df is None:
            df = self.df
        df=df.copy()
        results_d={}
        
        # random denormalized  
        df['entry']=random.choices([0,1],k=len(df))
        df['exit']=random.choices([0,1],k=len(df))
        _,r,_=self.backtest(df=df,money=money,entry_signal='entry',exit_signal='exit',price_col=price_col,normalize=False)
        results_d['random_denorm']=r
        # random normalized 
        _,r,_=self.backtest(df=df,money=money,entry_signal='entry',exit_signal='exit',price_col=price_col,normalize=True)
        results_d['random_norm']=r
        # hodl denormalized 
        df['entry']=1
        df['exit']=0
        _,r,_=self.backtest(df=df,money=money,entry_signal='entry',exit_signal='exit',price_col=price_col,normalize=False)
        results_d['hodl_denorm']=r
        # hodl normalized 
        _,r,_=self.backtest(df=df,money=money,entry_signal='entry',exit_signal='exit',price_col=price_col,normalize=True)
        results_d['hodl_norm']=r
         
        # linreg denormalized and normalized 
        X = df.index.tolist()
        y = df['close'].tolist()
        y_norm=(df['close']/self.normfun(df['close']) ).tolist()
        slope, intercept=np.polyfit(X,y,1)
        start_pa=X[0]
        end_pa=X[-1]
        r_linreg=(intercept+slope*end_pa) / (intercept+slope*start_pa)
        results_d['linreg']=r_linreg
        slope, intercept=np.polyfit(X,y_norm,1)
        r_linreg_norm=(intercept+slope*end_pa) / (intercept+slope*start_pa)
        results_d['linreg_norm']=r_linreg_norm
        
        
        
        return results_d


    # backtests a strategy 
    def backtest(self, df=None, money=10000, entry_signal='entry', exit_signal='exit', price_col=['open', 'close']
                 ,normalize=True       # normalizes open / close price by its mean - kind of introduces lookahead since mean uses it 
                 ,fraction_buy=1    # % of input money to buy when signal = 1 
                 ,fraction_sell=1):    # % of input money in current amo price to sell when signal = 0 
        if df is None:
            df = self.df
        df=df.copy()
        
        fraction_money_buy=money*fraction_buy      # corresponding money amount to buy 
        fraction_money_sell=money*fraction_sell    # corresponding money amount to sell 
        money_zero=money                           # used for fraction calculation at the end 
        if normalize:
            df['close']=df['close']/self.normfun(df['close'])     #   df['close'].ewm(span=100).mean()  # ewm mean normalization
            df['open']=df['open']/self.normfun(df['open'])        #  df['open'].ewm(span=100).mean() 
        
        trades_df=pd.DataFrame(columns=['entry','exit','pa','profit','profit_perc','money','amo','value','side','portfolio','timestamp','comment','amo_made']) # trades df 
        trades_d={k:0 for k  in trades_df.columns} 

        good_trades,bad_trades=0,0
        amo = 0 # current amo of asset 
        save_trade=0 # bool for saving trades in df 
        trade_side='none'
        buy_price=1
        sell_price=1
        for _, row in df.iterrows():
            d = row.to_dict()
            if d[entry_signal] == self.buy_flg and money > 0:      # buying due to signal 
                trades_d['comment']=''
                buy_price=d[price_col[0]]               # buying at current price
                pa=buy_price
                save_trade,trade_side=1,1                            # saving this trade 
                if fraction_money_buy < money and fraction_buy!=1:          # buying a fraction or going all in 
                    buy_money = fraction_money_buy
                else:
                    buy_money = money                   
                buy_amount = buy_money  / d[price_col[0]] # corresponding amount of asset based on current price 
                logging.warning(f" buying amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")
                amo += buy_amount                      # updating held amo 
                money -= buy_money                     # updating held money 
                logging.warning(f" bought amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")

            if d[exit_signal] == self.sell_flg and amo > 0:      # selling due to signal 
                sell_price=d[price_col[1]]            # selling at current price
                pa=sell_price
                if sell_price> buy_price:    # updating peak asset price
                    trades_d['comment']=f'good trade'
                    good_trades+=1
                else:
                    bad_trades+=1
                    trades_d['comment']=f'bad trade'
                
                
                save_trade,trade_side=1,0
                fraction_amo=fraction_money_sell / d[price_col[1]]         
                if fraction_amo < amo and fraction_sell!=1:          
                    sell_amount = fraction_amo
                else:
                    sell_amount=amo
                logging.warning(f"selling amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")
                
                money += sell_amount * d[price_col[1]]
                amo -= sell_amount
                
                logging.warning(f"sold amo: {amo}, money: {money}, price data: {row[price_col].to_dict()}, entry: {row[entry_signal]}, exit: {row[exit_signal]}")
            if save_trade==1:
                save_trade=0
                
                trades_d['money']=money
                trades_d['amo']=amo
                trades_d['entry']=row[entry_signal]
                trades_d['exit']=row[exit_signal]
                trades_d['pa']=pa
                trades_d['amo_made']=buy_price/sell_price
                trades_d['value']=amo*row[price_col[1]]+money
                trades_d['side']=trade_side
                trades_d['timestamp']=row['timestamp']
                trades_df.loc[len(trades_df)]=trades_d
                #print(trades_df)
                #input('wait')


        if amo != 0:                                    # selling everything at the end  - shouldnt happen 
            money += amo * df.iloc[-1][price_col[1]]
            logging.warning(f"selling at the end : {amo}, money: {money}, price data: {df.iloc[-1].to_dict()} ")
            trades_d={k:0 for k  in trades_df.columns}
            trades_d['money']=money
            trades_d['amo']=amo
            trades_d['side']=0
            trades_d['value']=money
            trades_d['timestamp']=row['timestamp']
            trades_df.loc[len(trades_df)]=trades_d
            
        logging.warning(trades_df.to_csv())             # saving trades df 
        return money, money/money_zero*100,trades_df



def plot_df(df, top_chart_cols=['col1', 'col2'], bottom_chart_cols=['col1', 'col3']):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    # Top chart
    if top_chart_cols is not None:
        for col in top_chart_cols:
            ax[0].scatter(df.index, df[col], label=col)
        ax[0].legend()
        ax[0].set_title('Top Chart: ' + ', '.join(top_chart_cols))
    
    
    # Bottom chart
    if bottom_chart_cols is not None:
        for col in bottom_chart_cols:
            ax[1].scatter(df.index, df[col], label=col)
        ax[1].legend()
        ax[1].set_title('Bottom Chart: ' + ', '.join(bottom_chart_cols))
    
    plt.tight_layout()
    ax[0].grid(True,which="both",ls="-", color='0.65')
    ax[1].grid(True,which="both",ls="-", color='0.65')
    plt.show()

    


if __name__=='__main__':
    setup_logging()
    fr=1

        
    df=pd.read_csv('./data/signals_df.csv',sep='|').reset_index()
    q1,q2=0,1 
    start=len(df)*q1//1
    end=len(df)*q2//1
    df=df.loc[start:end]
    # normalize data 
    #df['close']=df['close']/df['close'].mean()
    #df['open']=df['open']/df['open'].mean()
    #df['high']=df['high']/df['high'].mean()
    #df['low']=df['low']/df['low'].mean()
    #df['wave_signal']=df['wave_signal'].astype(int)
    #df['exit_signal']=abs(df['wave_signal']-1).astype(int)
    
    money,r,_=signals().backtest(df,entry_signal='wave_signal',exit_signal='wave_signal',normalize=True,fraction_buy=1, fraction_sell=fr )
    print(f'fr: {fr} r: {r}')
#    exit(1)
    d=signals().backtest_baseline(df)
    print(d)
#    print(f'fr: {fr} r: {r}')

    
