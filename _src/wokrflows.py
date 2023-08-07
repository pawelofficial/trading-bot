from bot.coinbase_api import * 
from bot.indicators import * 
from rich.traceback import install
install()


def wf__download_data():
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)
    fp=ca.bulk_download_data(
                            start_dt='01-01-2023'
                           ,end_dt='01-02-2023'
                           ,granularity=60*5
                           )
    return fp 
    
def wf__prep_data(fp=None):    
    i=indicators(fp=fp)
    i.aggregate_df(inplace=True)   
    i.bucketize_df()
    q_df,q_fp=i.dump_df(cols=i.quantile_columns,fname='quantiles_df')
    i_df,i_fp=i.dump_df(cols=i.basic_columns,fname='indicators_df')
    return q_df,q_fp,i_df,i_fp
    
    
    
    
fp=wf__download_data()
q_df,q_fp,i_df,i_fp=wf__prep_data(fp=fp)