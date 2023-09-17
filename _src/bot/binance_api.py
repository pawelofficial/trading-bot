from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import json 
from utils import setup_logging,log_stuff
import logging
import pandas as pd 
import random 
import numpy as np 
import datetime
import threading 
import time 
import queue 

# mock trade desk 

# live feed into a csv ? 
# model evaluating csv ? 
    # evaluating when new candle comes in ? 
# model training on csv ? 
    # training when new candle comes in ?
    

#def threaded(fn):
#    def wrapper(*args, **kwargs):
#        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
#        thread.daemon=True            # stopping abruptly when main program stops
#        thread.start()
#        return thread  # Optionally return the thread if you need to interact with it later
#    return wrapper

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        stop_event = threading.Event()
        thread.daemon=True            # stopping abruptly when main program stops
        thread.start()
        return thread,stop_event  # Optionally return the thread if you need to interact with it later
    return wrapper


class binance_api:
    def __init__(self,secrets_fp=None) -> None:
        setup_logging(name='binance',mode='w')
        log_stuff(msg='starting binance api ')
        self.secrets_fp=secrets_fp or './secrets/secrets.json'
        self.api_key,self.api_secret=self.get_secrets()
        self.Client=Client
        self.client=self.Client(self.api_key,self.api_secret)
        self.crypto_symbol='ETH'
        self.denom_symbol='BUSD'
        self.symbol=self.crypto_symbol+self.denom_symbol
        self.trade_amo_d={ 'trade_amo':None 
                          ,'trade_quant':None 
                          ,'trade_amo_p':0.2       # percentage per trade 
                          ,'trade_quant_p':1      # percentage per trade 
                          }
        
        self.precision=4
        self.dollar_amo=50                                # starting dollar  
        self.quantity=0                                    # starting quantity  

        self.balance={'quantity': self.quantity
                      ,'dollar_amo':self.dollar_amo
                      ,'value':self.dollar_amo+self.quantity* self.check_prices(symbol=self.symbol)[0] }

        self.cur_price=None # updating price only in check price 
        self.cur_ts = lambda : self.client.get_server_time()['serverTime']
        self.dollar_slippage=1
        self.trade_df=pd.DataFrame(columns=['symbol','quantity','price','ts','dollar_amo','change','side','comment'])
        self.balance_df=pd.DataFrame(columns=['quantity','dollar_amo','value', 'comment'])
        self.balance_df.loc[len(self.balance_df)]=self.balance
        self.rand=random.randint(-10,10)/100+1
        self.clock_probe_time=1
        self.q=queue.Queue(maxsize=1)
        #self.q.put(None) # put none to queue 
    
    def check_queue(self):
        return self.q.get(timeout=self.clock_probe_time * 10)
        
    def get_secrets(self):
        with open(self.secrets_fp) as f:
            secrets=json.load(f)
        return secrets['binance_api_key'],secrets['binance_api_secret']
    
    def check_prices(self,symbol=None):
        ###prices = self.client.get_all_tickers()
        ###price=None
        ###if symbol is not None:
        ###    price=[p for p in prices if p['symbol']==symbol][0]['price']
        ####return 1,{}
        prices=None
        price=100
        self.rand=random.randint(-10,10)/100+1
        self.cur_price=np.round(float(price),5)*self.rand
        return  self.cur_price,prices
    
    def get_amount(self,symbol=None,fake=False):
        return 0.001

    def fake_market_buy(self,dollar_amo=None,symbol=None):
        # log what is happening 
        log_stuff(msg='starting fake market buy ',dollar_amo=dollar_amo,symbol=symbol,balance=self.balance)                                                 
        # rewrite variables 
        symbol=symbol or self.symbol     
        self.check_prices(symbol=symbol)
                                                                                                                           
        value=self.cur_price*self.balance['quantity'] + self.balance['dollar_amo']
        dollar_amo= dollar_amo or self.trade_amo_d['trade_amo'] or self.trade_amo_d['trade_amo_p']*value       
        # do logic 
        com='OK'
        log_stuff(msg='attempting fake market buy ',dollar_amo=dollar_amo,symbol=symbol,balance=self.balance)
        if dollar_amo > self.balance['dollar_amo'] or self.balance['dollar_amo']==0:                                         # slippage
            log_stuff(msg=' not enough money to buy ',dollar_amo=dollar_amo,balance=self.balance)
            com='not enough money to buy'
            return 
        # execute action 
        
        quantity=dollar_amo/float(self.cur_price)*10**self.precision
        quantity=int(quantity)/10**self.precision
        change=dollar_amo -quantity*float(self.cur_price)
        if change<0:
            print('uh oh math is difficult')
        ###order = self.client.create_test_order(
        ###    symbol=symbol,
        ###    side=self.Client.SIDE_BUY,
        ###    type=self.Client.ORDER_TYPE_MARKET,
        ###    quantity=quantity)
        trade_d={'symbol':symbol,'quantity':quantity,'price':self.cur_price,'ts':self.cur_ts(),'dollar_amo':dollar_amo, 'change':change,'side':'BUY','comment':com}
        self.trade_df.loc[len(self.trade_df)]=trade_d
        self.save_df(df=self.trade_df,name='trade_df.csv')
        self.update_balance(trade_d)
        # log what happened 
        log_stuff(msg='ending fake market buy ',trade_d=trade_d,balance=self.balance)
        return trade_d 

    def update_balance(self,trade_d):
        log_stuff(msg=f'updating balance',trade_d=trade_d,balance=self.balance )
        if trade_d is None: # if no trade_d provided update only balance 
            self.balance['value']=self.balance['dollar_amo']+self.balance['quantity']*self.cur_price
            return 
        if trade_d['side']=='BUY':
            self.balance['dollar_amo']-=trade_d['dollar_amo'] - trade_d['change']
            self.balance['quantity']+=trade_d['quantity']
        elif trade_d['side']=='SELL':
            self.balance['dollar_amo']+=trade_d['dollar_amo']
            self.balance['quantity']-=trade_d['quantity']
        log_stuff(msg=f'balance updated',  balance = self.balance)
        tmp_d={'side':trade_d['side'], 'trade_d':trade_d}
        self.balance['value']=self.balance['dollar_amo']+self.balance['quantity']*self.cur_price
        log_stuff(msg=f'updated  balance',balance= self.balance)
        
        self.balance_df.loc[len(self.balance_df)]={**self.balance}
        self.save_df(df=self.balance_df,name='balance_df.csv')
        
        return 
    def save_df(self,df,name):
        fp=f'./data/binance_api/{name}' 
        numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        df[numeric_cols] = df[numeric_cols].round(self.precision)
    
        df.to_csv(fp,index=False,header=True,sep='|' )
        return

    def fake_market_sell(self,quantity=None,symbol=None):
        # log what is happening 
        log_stuff(msg='starting fake market SELL ',quantity=quantity,symbol=symbol,balance=self.balance)
        symbol=symbol or self.symbol  
        self.check_prices(symbol=symbol)
        
        # rewrite variables 
                                                                                                                              
        quantity=quantity or self.trade_amo_d['trade_quant'] or self.balance['quantity']*self.trade_amo_d['trade_quant_p']
        # do logic 
        log_stuff(msg='attempting fake market SELL ',quantity=quantity,symbol=symbol,balance=self.balance)
        com='OK'
        if quantity > self.balance['quantity'] or self.balance['quantity']==0:
            log_stuff(msg=' not enough crypto to sell ',quantity=quantity,balance=self.balance['quantity'])
            com='not enough crypto to sell'
            return
            
        # execute action 

        dollar_amo=quantity*float(self.cur_price)
        ###order = self.client.create_test_order(
        ###    symbol=symbol,
        ###    side=self.Client.SIDE_SELL,
        ###    type=self.Client.ORDER_TYPE_MARKET,
        ###    quantity=quantity)
        

        trade_d={'symbol':symbol,'quantity':quantity,'price':self.cur_price,'ts':self.cur_ts(),'dollar_amo':dollar_amo,  'change':0,'side':'SELL','comment':com}
        self.trade_df.loc[len(self.trade_df)]=trade_d
        self.update_balance(trade_d)
        self.save_df(df=self.trade_df,name='trade_df.csv')
        log_stuff(msg='ending fake market sell ',trade_d=trade_d,balance=self.balance)
        
        
    # clock checking if it's time to trade or not 
    @threaded
    def candle_clock(self, timescale=60,tradescale=5):                                                           # timescale - how often to trade, tradescale - how close to timescale trading should happen
        timedelta=lambda t1,t2: (t1-t2).total_seconds()
        while True:
            time.sleep(self.clock_probe_time)
            uts=int(datetime.datetime.now().timestamp())                                                         # unixtimestamp time -> 1681247182591 
            #uts=int(self.client.get_server_time()['serverTime'])//1000                                           # binance server time 
            ts=datetime.datetime.fromtimestamp(uts)                                                              # real time from uts -> 2023-09-17 09:16:11
            scale_ts=datetime.datetime(ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second//timescale*timescale) # scale time now     -> 2023-09-17 09:16:00
            next_scale_ts=scale_ts+datetime.timedelta(seconds=timescale)                                         # next scale         -> 2023-09-17 09:16:30
            delta=timedelta(ts,next_scale_ts)
            if delta>-1*tradescale:                                                                              # if we're close enough to next scale then it is time to trade 
                self.time_to_trade=True 
                #log_stuff(msg=f'time to trade',scale_ts=scale_ts,ts=ts,next_scale_ts=next_scale_ts,delta=delta  )
            else:
                self.time_to_trade=False 
                #log_stuff(msg=f'no time to trade',scale_ts=scale_ts,ts=ts,next_scale_ts=next_scale_ts,delta=delta  )
            self.q.put(self.time_to_trade)






if __name__=='__main__':
    b=binance_api()

    
    thread,stop_event = b.candle_clock()
    while True:
        time.sleep(1)
        result=b.check_queue()
        print(result)


#    N= 5000
#    n=0
#    while n<N: 
##        d=b.fake_market_sell()
#        
#        n+=1
#        if b.rand<1:
#            print('buying')
#            d=b.fake_market_buy()
#        else:
#            print('selloing')
#            d=b.fake_market_sell()
#
#
#    print(d)
#    d=b.fake_market_buy()
    
#    log_stuff(df=b.trade_df)
#    print(d)