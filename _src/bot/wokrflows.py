from coinbase_api import * 
from indicators import * 
from signals import * 
from rich.traceback import install
install()


def wf__download_data():
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)
    fp=ca.bulk_download_data(
                            start_dt='01-01-2023'
                           ,end_dt='01-02-2023'
                           ,granularity=60*5
                           ,fname='data_1M.csv'
                           )
    return fp 
    
def wf__prep_data(fp=None):    
    i=indicators(fp=fp)
    i.aggregate_df(inplace=True)   
    i.bucketize_df()
    q_df,q_fp=i.dump_df(cols=i.quantile_columns,fname='quantiles_df')
    i_df,i_fp=i.dump_df(cols=i.basic_columns,fname='indicators_df')
    
    signal,s_df=signals().signal_wave2(df=i_df)
    s_df,s_fp=signals().dump_df(df=s_df,fname='signals_df')
    
    return q_df,q_fp,i_df,i_fp,s_df,signal
    
    
    
if __name__=='__main__':
    fp=wf__download_data()
    q_df,q_fp,i_df,i_fp,s_df,signal=wf__prep_data(fp=fp)