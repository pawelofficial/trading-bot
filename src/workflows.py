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
def wf__process_df(df_fps ='./data/data.csv',scale=15,out_fps='./src/data/math_df.csv'):
    mth=myMath()
    mth.read_csv(filepath=df_fps)
    math_df=mth.aggregate(df=mth.math_df, scale =scale, src_col='timestamp')
    df=make_math_df(math_df.copy(deep=True))
    print(df.columns)
    df.rename(columns={f'ts-{scale}':'timestamp', 'rsi-ema':'rsi'}, inplace=True)
    print(df.columns)
    print('dumping')
    mth.dump_csv(df=df,filename=out_fps)
    
    
def wf__make_indicators(df_fps='./src/data/data.csv',out_fps='./src/data/indicators_df.csv'):
    u=Utils()
    df=u.read_csv(fps=df_fps)
    df['index']=df.index
    i=indicators(df=df)
    
    for fun in i.funs_d.values():
        fun()
    print(i.df.columns)
    
    
if __name__=='__main__':
    wf__make_indicators()