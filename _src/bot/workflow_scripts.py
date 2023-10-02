from coinbase_api import * 
from indicators import * 
from signals import * 
from torch_model4 import *
from rich.traceback import install
from my_pgsql import * 
import utils
import sys 
#from torch_model4 import Network
install()

# continous feed data.csv  1min timeframe 
# 


# 1st thing - download data 
def wf__download_data():
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)
    fp=ca.bulk_download_data(
                            start_dt='16-09-2023' # DD-MM-YYYY
                           ,end_dt='26-09-2023'   # DD-MM-YYYY
                           ,granularity=60
                           ,fname='data.csv'
                           )
    return fp 
    
# 2nd thing - make features for ML 
def wf__prep_data(fp=None):    
    i=indicators(fp=fp)
    i.aggregate_df(inplace=True)   
    i.df.to_csv('./data/agg_df.csv',sep='|')
    i.bucketize_df()
    q_df,q_fp=i.dump_df(cols=i.quantile_columns,fname='quantiles_df')
    i_df,i_fp=i.dump_df(cols=i.basic_columns,fname='indicators_df')
    
    signal,s_df=signals().signal_wave2(df=i_df)
    s_df,s_fp=signals().dump_df(df=s_df,fname='signals_df')
    
    return q_df,q_fp,i_df,i_fp,s_df,signal
    
    

    

    
# 3rd thing - test your strategy / ml 
def wf__evaluate_model():
    cols=['open','close','high','low','timestamp','wave_signal','model_signal']
    signals_fp='./data/signals_df.csv'
    quantiles_fp='./data/quantiles_df.csv'
    model_fp='./models/wave_models/wave_loop.pth'
    signals_df=utils.read_df(signals_fp)
    quantiles_df=utils.read_df(quantiles_fp)
    model=utils.read_model(model_fp,quantiles_df,Network)
    df=utils.evaluate_model(model,quantiles_df,signals_df )
    df['exit_signal']=abs(1-df['model_signal'])
    return df 
#    money,r,trades_df=signals().backtest(df,entry_signal='model_signal',exit_signal='exit_signal',normalize=False,fraction_buy=1, fraction_sell=1 )
#    
#    additional_sers=[
#        (0,df['timestamp'],df['close'],'--','.','magenta' )
#        ,(1,trades_df['timestamp'],trades_df['value'],'','v','magenta' )
#    ]
#    
#    top_cols=None
#    bot_cols=None 
#    utils.plot_two_dfs(top_df=trades_df,top_chart_cols=top_cols,top_index='timestamp'
#                       ,bot_df=trades_df,bottom_chart_cols=bot_cols,bot_index='timestamp'
#                       ,additional_sers= additional_sers 
#                       )
#    
#    utils.write_df(df=trades_df,name='trades_df')
#    return r

    
def wf__live_pg_write():
    l=utils.setup_logging2(name='wf__live_pg_write')
    utils.log_stuff2(l,msg='wf__live_pg_write log ')
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)
    gen=ca.yield_last_candle()
    
    p=mydb()
    p.execute_dml('truncate table live_data')
    while True:
        c=next(gen)
        data=c['parsed_data']
        df=pd.DataFrame(data)
        utils.log_stuff2(l,msg=f'writing df {df}')
        p.write_df(df=df,table='live_data',if_exists='append')
        time.sleep(60)
    


if __name__=='__main__':
    wf__live_pg_write()
    if len(sys.argv)>1:
        if sys.argv[1] == "wf__live_pg_write":
            wf__live_pg_write()
        

