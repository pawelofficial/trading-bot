import os 

from coinbase_api import * 
from mymath import * 
from indicators import * 
from utils import *
from myfuns import pca_plot
import logging 
logging.basicConfig(filename='./src/logs/workflows.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')

# downloads data from coinbase 
def wf__download_by_dates(start_date='01-01-2022',end_date='03-01-2022',config_fps='./src/secrets/secrets.json',out_fps='./src/data/'):
    logging.warning(f'starting download_by_dates  {start_date} {end_date}')
    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
    api=utils.init(config_filepath=os.path.abspath(config_fps))
    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
    fps=utils.download_by_dates(api=api,start_date=start_date,end_date=end_date,path=os.path.abspath(out_fps))
    logging.warning(f'finished download_by_dates  {fps}')
    return fps

   # aggregates df to different timeframe, ads fk column and dumps 
def wf__aggregate_df(df_fps ='./data/data.csv'
                     ,scale=15
                     ,out_fps='./src/data/math_df.csv'
                     ,scale_key=None# ads additional scale key for joining with other timeframes 
                     , dump_df=False
                     ,df =None ):  # provide df if you want to use it instead of reading from file
    logging.warning(f'starting aggregate_df')
    mth=myMath()
    if df is None:
        logging.warning(f'reading df  {df_fps}')
        mth.read_csv(filepath=df_fps)
    else:
        print('using provided df')
        mth.math_df=df
    logging.warning(f'original df len {len(mth.math_df)}')
    math_df=mth.aggregate(df=mth.math_df, scale =scale, src_col='timestamp')
    logging.warning(f'aggregated df len {len(math_df)}')
    if scale_key is not None:
        logging.warning(f'adding scale key {scale_key}')
        math_df[f'ts-{scale_key}']=mth.calculate_fun(df=math_df,fun_name='floor_datetime',str_col='ts-30',scale=scale_key,inplace=False)
    math_df.rename(columns={f'ts-{scale}':'timestamp'}, inplace=True)
    if dump_df:
        logging.warning(f'dumping df {out_fps}')
        mth.dump_csv(df=math_df,filename=out_fps)
    return math_df 


def wf__make_quantiles_df(input_df,nlq_number=None,nlq_steepness=None,nlq_accuracy=None):
    logging.warning('starting make_quantiles_df')
    logging.warning(f'original df columns count {len(input_df.columns)}')
    u=Utils()
    input_df['index']=input_df.index
    i=indicators(df=input_df)
    if nlq_number is not None:  
        i.nlq_number=nlq_number
    if nlq_steepness is not None:  
        i.nlq_steepness=nlq_steepness
    if nlq_accuracy is not None:
        i.nlq_accuracy=nlq_accuracy
    logging.warning(f'nlq values {i.nlq_number} {i.nlq_steepness} {i.nlq_accuracy}')
    
    for fun, params in i.funs_d.values():  # calculate indicators  
        fun(**params)
    base_columns=i.df.columns              # basic columns and indicator columns 
    for col in i.q_cols():                 # calculate quantiles columns 
        i.bucketizeme(df=i.df,col=col) 
    quantile_columns=[col for col in i.df.columns if col not in base_columns] # quantile columns
    logging.warning('finished make_quantiles_df')
    logging.warning(f'quantiles columns count {len(quantile_columns)}  base columns count {len(base_columns)} ')
    logging.warning(f'quantiles columns {quantile_columns}  \n base columns {base_columns} ')
    
    return i.df[quantile_columns],i.df[base_columns] # return quantile df and base df
        
    

    
    
def wf__make_indicators(df_fps='./src/data/data.csv',out_fps='./src/data/quantiles_df.csv',input_df=None,dump=False):
    u=Utils()
    if input_df is not None:
        df=input_df
    else:
        df=u.read_csv(fps=df_fps)
    df['index']=df.index
    i=indicators(df=df)
    for fun, params in i.funs_d.values():
        fun(**params)
    cur_cols=i.df.columns
    for col in i.q_cols():
        i.bucketizeme(df=i.df,col=col)
    logit_cols=[col for col in i.df.columns if col not in cur_cols]
    logit_cols.append('green')
    candle_size=(i.df['close'] - i.df['open']).abs()
    mean_candle_size = candle_size.mean()
    std_candle_size = candle_size.std()
    i.df['next_close'] = i.df['close'].shift(-1)
    i.df['next_open'] = i.df['open'].shift(-1)
    #i.df['green'] = (i.df['next_close'] >= i.df['next_open']).astype(int)
    n=1
    i.df['green'] = (((i.df['next_close'] - i.df['next_open']) > mean_candle_size + n*std_candle_size) & (i.df['next_close'] > i.df['next_open'])).astype(int)
    if dump:
        u.dump_csv(df=i.df[logit_cols] ,fps=out_fps)
    extra_cols=[c for c in i.df.columns if c not in logit_cols]
    extra_cols.append('green')
    edf=pd.DataFrame({}) # extra df 
    edf[extra_cols]=i.df[extra_cols]
    i.df.drop(columns=extra_cols,inplace=True)
    return i.df,edf,logit_cols,extra_cols


def wf__evaluate_model(data_fps='./src/data_backups/BTC-USD2022-01-01_2023-01-01.csv'
                       ,scales=[15,60]):
    ts_fast=f'ts-{scales[0]}'
    ts_slow=f'ts-{scales[1]}'
    mth=myMath()
    data_df=pd.read_csv(data_fps)
    
    fast_df=mth.aggregate(df=data_df, scale =scales[0], src_col='timestamp')
    slow_df=mth.aggregate(df=data_df, scale =scales[1], src_col='timestamp')
    fast_df[ts_slow]=mth.calculate_fun(df=fast_df,fun_name='floor_datetime',str_col=ts_fast,scale=scales[1],inplace=False)    
    qdf_fast,edf_fast,logit_cols_fast,extra_cols_fast=wf__make_indicators(input_df=fast_df)
    qdf_fast.rename(columns={col: 'fast_' + col for col in qdf_fast.columns}, inplace=True)
    edf_fast.rename(columns={col: 'fast_' + col for col in qdf_fast.columns}, inplace=True)

    qdf_slow,edf_slow,logit_cols_slow,extra_cols_slow=wf__make_indicators(input_df=slow_df)
    qdf_slow.rename(columns={col: 'slow_' + col for col in qdf_slow.columns}, inplace=True)
    edf_slow.rename(columns={col: 'slow_' + col for col in qdf_slow.columns}, inplace=True)
    
    qdf_fast['fk']=edf_fast[ts_slow]
    qdf_slow['pk']=edf_slow[ts_slow]

    qdf=qdf_fast.merge(qdf_slow,how='inner',left_on='fk',right_on='pk')

    print(len(qdf.columns))
    print(len(qdf_fast.columns))
    print(len(qdf_slow.columns))
    
    print(len(qdf))
    print(len(qdf_fast))
    print(len(qdf_slow))
    qdf.drop(columns=['fk','pk'],inplace=True)
    qdf_fast.drop(columns=['fk'],inplace=True)
    s='3months_s15_a5_N100'
    qdf.to_csv(f'./src/data/qdf_{s}.csv',index=True)
    edf_fast.to_csv(f'./src/data/edf_{s}.csv',index=True)
    #qdf=pd.read_csv('./src/data_backups/indicators_BTC-USD2022-01-01_2022-02-03.csv')
    ##qdf['labels']=edf_fast['green']
    #pca_plot(df=qdf,labels=edf_fast['green'],n_components=2)
    
if __name__=='__main__':
    df=pd.read_csv('./src/data/raw/data.csv').iloc[:500]
    wf__make_quantiles_df(input_df=df)
    
    #wf__download_by_dates(start_date='01-01-2023',end_date='01-02-2023',out_fps='./src/data/raw/')
    #wf__evaluate_model(data_fps='./src/data/raw/BTC-USD2023-01-01_2023-03-01.csv')
    exit(1)
    dataset='BTC-USD2022-01-01_2022-01-03' # 2 days 
    #dataset='BTC-USD2022-01-01_2022-02-03' # 3 months
    #dataset='BTC-USD2022-01-01_2023-01-01'
    #wf__download_by_dates(start_date='01-01-2022',end_date='03-01-2022',out_fps='./src/data/')
    df_30=wf__aggregate_df(df_fps =f'./src/data_backups/{dataset}.csv',scale=30,out_fps='./src/data/data.csv',scale_key=60)
    #df_1h=wf__aggregate_df(df_fps =f'./src/data_backups/{dataset}.csv',scale=60,out_fps='./src/data/data.csv')
    #wf__make_indicators(out_fps='./src/data/indicators_BTC-USD2022-01-01_2022-02-03.csv')
    print(df_30.head())
    #print(df_1h.head())