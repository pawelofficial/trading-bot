from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import json 
from utils import setup_logging,log_stuff
import logging
import pandas as pd 






# mock trade desk 

# live feed into a csv ? 
# model evaluating csv ? 
    # evaluating when new candle comes in ? 
# model training on csv ? 
    # training when new candle comes in ?
    


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
        self.dollar_amo=25                                 # trade dollar amo 
        self.quantity=0                                    #
        self.balance={'quantity': self.quantity,'dollar_amo':self.dollar_amo}
         
        self.cur_ts = lambda : self.client.get_server_time()['serverTime']
        self.dollar_slippage=1
        self.trade_df=pd.DataFrame(columns=['symbol','quantity','price','ts','dollar_amo','change','side'])
        self.balance_df=pd.DataFrame(columns=['quantity','dollar_amo','side', 'trade_d'])
        self.balance_df.loc[len(self.balance_df)]=self.balance

        
    def get_secrets(self):
        with open(self.secrets_fp) as f:
            secrets=json.load(f)
        return secrets['binance_api_key'],secrets['binance_api_secret']
    
    def check_prices(self,symbol=None):
        prices = self.client.get_all_tickers()
        price=None
        if symbol is not None:
            price=[p for p in prices if p['symbol']==symbol][0]['price']
        #return 1,{}
        return  price,prices
    
    def get_amount(self,symbol=None,fake=False):
        return 0.001

    def fake_market_buy(self,dollar_amo=None,symbol=None):
        symbol=symbol or self.symbol
        dollar_amo=dollar_amo or self.dollar_amo
        dollar_amo=dollar_amo-self.dollar_slippage
        log_stuff(msg='starting fake market buy ',dollar_amo=dollar_amo,symbol=symbol)
        if dollar_amo > self.balance['dollar_amo']:
            log_stuff(msg=' not enough money to buy ',dollar_amo=dollar_amo,balance=self.balance['dollar_amo'])
            return 
        
        price=self.check_prices(symbol=symbol)[0]
        quantity=dollar_amo/float(price)
        quantity=round(quantity,4)
        change=dollar_amo -quantity*float(price)
        order = self.client.create_test_order(
            symbol=symbol,
            side=self.Client.SIDE_BUY,
            type=self.Client.ORDER_TYPE_MARKET,
            quantity=quantity)
        
        trade_d={'symbol':symbol,'quantity':quantity,'price':price,'ts':self.cur_ts(),'dollar_amo':dollar_amo, 'change':change,'side':'BUY'}
        self.trade_df.loc[len(self.trade_df)]=trade_d
        log_stuff(msg='ending fake market buy ',trade_d=trade_d)
        self.update_balance(trade_d)
        return trade_d 

    def update_balance(self,trade_d):
        log_stuff(msg=f'updating balance {self.balance}')
        if trade_d['side']=='BUY':
            self.balance['dollar_amo']-=trade_d['dollar_amo'] - trade_d['change']
            self.balance['quantity']+=trade_d['quantity']
        elif trade_d['side']=='SELL':
            self.balance['dollar_amo']+=trade_d['dollar_amo']
            self.balance['quantity']-=trade_d['quantity']
        log_stuff(msg=f'balance updated  {self.balance}')
        tmp_d={'side':trade_d['side'], 'trade_d':trade_d}
        
        self.balance_df.loc[len(self.balance_df)]={**self.balance,**tmp_d}
        print(self.balance_df)


    def fake_market_sell(self,quantity=None,symbol=None):
        log_stuff(msg=' fake market sell ',quantity=quantity,symbol=symbol)
        symbol=symbol or self.symbol
        quantity=quantity or self.balance['quantity']
        
        if quantity > self.balance['quantity']:
            log_stuff(msg=' not enough crypto to sell ',quantity=quantity,balance=self.balance['quantity'])
        
        price=self.check_prices(symbol=symbol)[0]
        dollar_amo=quantity*float(price)
        order = self.client.create_test_order(
            symbol=symbol,
            side=self.Client.SIDE_SELL,
            type=self.Client.ORDER_TYPE_MARKET,
            quantity=quantity)
        
        trade_d={'symbol':symbol,'quantity':quantity,'price':price,'ts':self.cur_ts(),'dollar_amo':dollar_amo,  'change':0,'side':'SELL'}
        self.trade_df.loc[len(self.trade_df)]=trade_d
        log_stuff(msg=' fake market sold ',trade_d=trade_d)
        self.update_balance(trade_d)
        

if __name__=='__main__':
    b=binance_api()
    b.balance['dollar_amo']=100
    
    
    d=b.fake_market_buy()
    d=b.fake_market_buy()
    d=b.fake_market_sell()

#    print(d)
#    d=b.fake_market_buy()
    
#    log_stuff(df=b.trade_df)
#    print(d)