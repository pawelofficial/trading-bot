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
def wf__download_data(write_to_pg=False
                      ,start_dt=None
                      ,end_dt=None):

    if start_dt is None:
        # dynamically set start_dt to week ago 
        start_dt=datetime.datetime.now()-datetime.timedelta(days=2)
        start_dt=start_dt.strftime('%d-%m-%Y')
    if end_dt is None:
        # dynamically set end_dt to tomorrow 
        end_dt=datetime.datetime.now()+datetime.timedelta(days=1)
        end_dt=end_dt.strftime('%d-%m-%Y')
    
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)    
    df,fp=ca.bulk_download_data(
                            start_dt=start_dt # DD-MM-YYYY
                           ,end_dt=end_dt   # DD-MM-YYYY
                           ,granularity=60
                           ,to_df=True
                           #to_csv=True,fname='raw_data.csv'
                           )

    if write_to_pg:
        p=mydb()
        p.execute_dml('truncate table historical_data')
        p.execute_dml('truncate table live_data')              # be careful with this one 
        p.write_df(df=df,table='historical_data',if_exists='append',deduplicate_on='epoch')
    return fp 
    

def wf__prep_data():
    p=mydb()
    df=p.execute_select('select  start_time as timestamp, open, close, low, high, volume from vw_agg5 order by start_epoch asc limit 300' )
    # cast open,close,low,high to float64
    df[['open','close','low','high','volume']]=df[['open','close','low','high','volume']].astype('float64')
    
    # bucketize data 
    i=indicators(df=df)
    i.bucketize_df() 
    
    # drop load quantiles table - drop load due to changing structure of this table :) 
    p.execute_dml('drop table quantiles')
    ddl=p.df_to_ddl(df=df[i.basic_columns],table_name='quantiles',extra_cols=[('ar','integer[]')  ])
    p.execute_dml(ddl)
    p.write_df_array(df=i.df,tgt_ar_col='ar',tbl='quantiles',df_ar_cols=i.quantile_columns,df_base_cols=i.basic_columns)

    # compute and  write signals - basic + signal columns 
    p.execute_dml('drop table signals')
    signal,s_df=signals().signal_wave2(df=i.df[i.basic_columns])
    ddl=p.df_to_ddl(df=s_df,table_name='signals' )
    p.execute_dml(ddl)
    p.write_df(df=s_df,table='signals',if_exists='append')

    
    
# 2nd thing - make features for ML 
def wf__prep_data_old(fp=None,write_to_pg=True):    
    i=indicators(fp='./data/data.csv')
    i.aggregate_df(inplace=True,scale=5)   
    i.df.to_csv('./data/agg_df.csv',sep='|')
    i.bucketize_df()
    q_df,q_fp=i.dump_df(df=i.df,cols=i.quantile_columns,fname='quantiles_df')
    
    i.basic_columns=['timestamp','open','close','low','high','volume']
    i_df,i_fp=i.dump_df(cols=i.basic_columns+i.fun_columns,fname='indicators_df')

    signal,s_df=signals().signal_wave2(df=i.df[i.basic_columns])
    s_df,s_fp=signals().dump_df(df=s_df,  fname='signals_df')


    
    

    if write_to_pg:
        p=mydb()
        #ar_columns=[x for x in q_df.columns.tolist() if 'q' in x]
        p.execute_dml('truncate table quantiles')
        df_base_cols=['timestamp','open','close','low','high','volume']
        print(len(i.quantile_columns))
        p.write_df_array(df=i.df,tgt_ar_col='ar',tbl='quantiles',df_ar_cols=i.quantile_columns,df_base_cols=df_base_cols)
        
        p.execute_dml('truncate table signals')
        signal_columns=[x for x in s_df.columns.tolist() if 'q' not in x]
        p.write_df(df=s_df,table='signals',if_exists='append',cols_list=signal_columns)

    
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

    #df=utils.plot_evaluate_model(model,quantiles_df,signals_df )
    df['exit_signal']=abs(1-df['model_signal'])
    msk=df['model_signal']==1
    #print(df[msk])
    #print(len(df[msk]))
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
    #wf__download_data() # downloads data.csv with open,close,low,high,volume,timestamp on 1m 
    wf__prep_data_old() # makes indicators.csv ( data with indicators) and quantiles.csv and signals.csv ( input to NN )
    # train model -> torch_model4.py 

#    wf__download_data()
#    wf__evaluate_model()
    exit(1)
#    wf__pgsql_evaluate()
#    exit(1)
#    wf__download_data()
    wf__prep_data()
#    wf__evaluate_model()
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