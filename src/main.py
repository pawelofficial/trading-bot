from coinbase_api import * 
from mymath import *

fp= os.path.abspath('./src/secrets/secrets.json')
mth=myMath()
utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
api=utils.init(config_filepath=fp)
scale=15
def prep_data(scale=scale):
    data_fp=os.path.abspath('./src/tmp/')
    
    if 1: # use csv 
        api.bulk_df=pd.read_csv('./src/tmp/bulk_df.csv')
        api.data_df=api.bulk_df.iloc[-30:]
        api.bulk_df=api.bulk_df.iloc[:-30]
    else:
        utils.download_last_n_days(api=api,n=1,path=data_fp) 
    
    columns=api.bulk_df.columns
    api.bulk_df['floor']=mth.calculate_fun(df=api.bulk_df,fun_name='floor_dt',str_col='timestamp',scale=scale)
    i=len(api.bulk_df)-1
    rows_to_drop=[]
    while True:
        d=api.bulk_df.iloc[i].to_dict()
        print(d)
        d['timestamp']=datetime.datetime.strptime(d['timestamp'],'%Y-%m-%d %H:%M:%S')
        if d['timestamp']!=d['floor']:
            rows_to_drop.append(i)
        else:
            break
        i-=1
    rows_to_drop.append(len(api.bulk_df)-1-len(rows_to_drop))
    rows_to_drop.reverse()
    remain_df=api.bulk_df.iloc[rows_to_drop]
    api.bulk_df=api.bulk_df.drop(rows_to_drop)
    api.data_df=pd.merge(remain_df,api.data_df,how='outer')
    api.bulk_df=api.bulk_df[columns]
    api.data_df=api.data_df[columns]
    


    agg_df=mth.aggregate(df=api.bulk_df,scale=scale,src_col='timestamp') # agg bulk df 
    
    agg_df.rename(columns={f'ts-{scale}':'timestamp'},inplace=True)
    agg_df['epoch']=agg_df['timestamp'].apply(lambda x: datetime.datetime.timestamp(x))
    agg_df=agg_df.sort_index(axis=1)
    api.data_df=api.data_df.sort_index(axis=1)
    api.data_df['timestamp'].astype('datetime64[ns]')
    return agg_df,api.data_df
    
def df_generator(df):
    for index, row in df.iterrows():
        yield row
    

if __name__=='__main__':
    agg_df,data_df=prep_data()
#    print(agg_df.tail())
#    print(data_df.head())
#    exit(1)

    gen_df=data_df.copy()
    data_df=pd.DataFrame(columns=data_df.columns)
    
    gen=df_generator(df=gen_df)
    print(agg_df.tail())
    while True:
        try:
            row=next(gen)                              
        except StopIteration:
            break
        data_df.loc[len(data_df)]=row       
        
        if len(data_df)==scale:
            row=mth.aggregate(df=data_df,scale=scale,src_col='timestamp').iloc[0].to_dict() # agg data_df
            row['epoch']=datetime.datetime.timestamp(row[f'ts-{scale}'])
            row['timestamp']=datetime.datetime.strftime(row[f'ts-{scale}'],'%Y-%m-%d %H:%M:%S')
            row={k:v for k,v in row.items() if k in agg_df.columns}
            agg_df.loc[len(agg_df)]=row
            data_df=pd.DataFrame(columns=agg_df.columns)


       # input('here')
        
    
    print(agg_df.tail())
    print(data_df.head())