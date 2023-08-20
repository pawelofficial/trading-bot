from coinbase_api import * 
from indicators import * 
from signals import * 
#from torch_model4 import *
from rich.traceback import install
import utils
install()

# 1st thing - download data 
def wf__download_data():
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)
    fp=ca.bulk_download_data(
                            start_dt='02-03-2023' # DD-MM-YYYY
                           ,end_dt='03-04-2023'   # DD-MM-YYYY
                           ,granularity=60*5
                           ,fname='data.csv'
                           )
    return fp 
    
# 2nd thing - make features for ML 
def wf__prep_data(fp=None):    
    i=indicators(fp=fp)
    i.aggregate_df(inplace=True)   
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
    model=utils.read_model(model_fp,quantiles_df)
    df=utils.evaluate_model(model,quantiles_df,signals_df )
   # df=df.iloc[:500]
    #print(df.columns)
    #print(df[cols].head(25))
#    money,r,df=signals().backtest(df,entry_signal='wave_signal',exit_signal='wave_signal',normalize=True,fraction_buy=1, fraction_sell=1 )
#    print(r)
    df['exit_signal']=abs(1-df['model_signal'])
    money,r,trades_df=signals().backtest(df,entry_signal='model_signal',exit_signal='exit_signal',normalize=False,fraction_buy=1, fraction_sell=1 )
    print(r)
    print(trades_df.tail(25))

    msk1=trades_df['comment']=='good trade'
    m1=trades_df[msk1]['amo_made'].sum()
    
    msk2=trades_df['comment']=='bad trade'
    m2=trades_df[msk2]['amo_made'].sum()
    
    m3=trades_df['amo_made'].sum()
    print(m1,m2,m1+m2,m3)

    s1=trades_df['value'][msk1].fillna(0)
    s2=trades_df['value'][msk2].fillna(0)
    top_cols=['open','close']
    bot_cols=None # ['open','close']
    
    s=df['model_signal']*df['open']
    additional_sers=[
        (0,df['timestamp'],s,'','v','magenta' )
        ,(1,trades_df['timestamp'],trades_df['value'],'','v','magenta' )
    ]
    
    utils.plot_two_dfs(top_df=df,top_chart_cols=top_cols,top_index='timestamp'
                       ,bot_df=df,bottom_chart_cols=bot_cols,bot_index='timestamp'
                       ,additional_sers= additional_sers 
                       )
#    utils.plot_two_dfs(top_df=trades_df,top_chart_cols=top_cols,top_index='timestamp'
#                       ,bot_df=df,bottom_chart_cols=bot_cols,bot_index='timestamp'
#                       ,additional_sers= additional_sers 
#                       )
    
    

    
if __name__=='__main__':

#    fp=wf__download_data()
#    q_df,q_fp,i_df,i_fp,s_df,signal=wf__prep_data(fp=fp)
#    wf__prep_data(fp)
    wf__evaluate_model()