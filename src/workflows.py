import os 

from coinbase_api import * 
from mymath import * 
from indicators import * 
from utils import *

# downloads data from coinbase 
def wf__download_by_dates(start_date='01-01-2022',end_date='03-01-2022',config_fps='./src/secrets/secrets.json',out_fps='./src/data/'):
    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
    api=utils.init(config_filepath=os.path.abspath(config_fps))
    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
    fps=utils.download_by_dates(api=api,start_date=start_date,end_date=end_date,path=os.path.abspath(out_fps))
    return fps


# aggregates and calculates stuff on a df 
def wf__aggregate_df(df_fps ='./data/data.csv',scale=15,out_fps='./src/data/math_df.csv'):
    mth=myMath()
    mth.read_csv(filepath=df_fps)
    math_df=mth.aggregate(df=mth.math_df, scale =scale, src_col='timestamp')
    math_df.rename(columns={f'ts-{scale}':'timestamp'}, inplace=True)
    mth.dump_csv(df=math_df,filename=out_fps)
    return math_df
    
    
def wf__make_indicators(df_fps='./src/data/data.csv',out_fps='./src/data/indicators_df.csv'):
    u=Utils()
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
    u.dump_csv(df=i.df[logit_cols] ,fps=out_fps)
    return i.df,logit_cols,i.df['green']


    
if __name__=='__main__':
    dataset='BTC-USD2022-01-01_2022-01-03' # 2 days 
    dataset='BTC-USD2022-01-01_2022-02-03' # 3 months
    #dataset='BTC-USD2022-01-01_2023-01-01'
    #wf__download_by_dates(start_date='01-01-2022',end_date='03-01-2022',out_fps='./src/data/')
    wf__aggregate_df(df_fps =f'./src/data_backups/{dataset}.csv',scale=15,out_fps='./src/data/data.csv')
    wf__make_indicators()