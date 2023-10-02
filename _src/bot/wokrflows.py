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
def wf__download_data(write_to_pg=True,start_dt=None,end_dt=None):

    if start_dt is None:
        # dynamically set start_dt to week ago 
        start_dt=datetime.datetime.now()-datetime.timedelta(days=7)
        start_dt=start_dt.strftime('%d-%m-%Y')
    if end_dt is None:
        # dynamically set end_dt to tomorrow 
        end_dt=datetime.datetime.now()+datetime.timedelta(days=1)
        end_dt=end_dt.strftime('%d-%m-%Y')
    
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)    

    fp=ca.bulk_download_data(
                            start_dt=start_dt # DD-MM-YYYY
                           ,end_dt=end_dt   # DD-MM-YYYY
                           ,granularity=60
                           ,fname='data.csv'
                           )

        
    if write_to_pg:
        p=mydb()
        df=pd.read_csv('./data/data.csv',sep='|')
        print(df)
        p.execute_dml('truncate table historical_data')
        #p.execute_dml('truncate table live_data')        # lmao dont truncate this 
        p.write_df(df=df,table='historical_data',if_exists='append')
    return fp 
    
    
# 2nd thing - make features for ML 
def wf__prep_data(fp=None):    
    i=indicators(fp='./data/data.csv')
    i.aggregate_df(inplace=True,scale=5)   
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
    #print(quantiles_df.shape) # 200 1200
    
    model=utils.read_model(model_fp,quantiles_df,Network)
    df=utils.evaluate_model(model,quantiles_df,signals_df )
    df['exit_signal']=abs(1-df['model_signal'])
    msk=df['model_signal']==1
    print(df[msk])
    print(len(df[msk]))
    df[msk].to_csv('./data/wf__evaluate_model_signals.csv',index=False)
    return df 

def wf__live_pg_write():
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)
    gen=ca.yield_last_candle()
    p=mydb()
    while True:
        c=next(gen)
        data=c['parsed_data']
        df=pd.DataFrame(data)
        print(df)    
        #p=mydb()        
        p.write_df(df=df,table='live_data',if_exists='append')
        time.sleep(60)

# gets data from pgsql, dumps intermediate features to csvs and runs the model on csvs 
def wf__pgsql_evaluate():
    p=mydb()
    df=p.execute_select('select * from vw_agg5 order by start_epoch asc' )
    # cast open,close,low,high to float64
    df[['open','close','low','high','volume']]=df[['open','close','low','high','volume']].astype('float64')
    df.to_csv('./data/pgsql_data.csv',index=False)
    i=indicators(df=df)
    i.bucketize_df()
    quantiles_df,quantiles_fp=i.dump_df(cols=i.quantile_columns,fname='quantiles_df')
    i_df,i_fp=i.dump_df(cols=i.basic_columns,fname='indicators_df')
    signal,s_df=signals().signal_wave2(df=i_df)
    signals_df,signals_fp=signals().dump_df(df=s_df,fname='signals_df')
    signals_fp='./data/signals_df.csv'
    quantiles_fp='./data/quantiles_df.csv'
    model_fp='./models/wave_models/wave_loop.pth'
    signals_df=utils.read_df(signals_fp)
    quantiles_df=utils.read_df(quantiles_fp)
    model=utils.read_model(model_fp,quantiles_df,Network)
    df=utils.evaluate_model(model,quantiles_df,signals_df )
    df['exit_signal']=abs(1-df['model_signal'])
    msk=df['model_signal']==1
    df[msk].to_csv('./data/pgsql_signals.csv',index=False)
    print(df[msk])
    print(len(df[msk]))



if __name__=='__main__':
    wf__pgsql_evaluate()
    exit(1)
    wf__download_data()
    wf__prep_data()
    wf__evaluate_model()
    exit(1)
    wf__pgsql_evaluate()
    exit(1)
#
#    fp=wf__download_data()
    
#    wf__prep_data(fp)
    df=wf__evaluate_model()
#    msk=df['model_signal']==1
#    print(df[msk])
##    q_df,q_fp,i_df,i_fp,s_df,signal=wf__prep_data(fp=fp)
#    wf__prep_data(fp)
#    wf__evaluate_model()