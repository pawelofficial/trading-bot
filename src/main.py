from coinbase_api import * 

def get_data():
    fp= os.path.abspath('./src/secrets/secrets.json')
    data_fp=os.path.abspath('./src/tmp/')
    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
    api=utils.init(config_filepath=fp)
    utils.download_last_n_days(api=api,n=1,path=data_fp) # download a lot of data here 
    print(max(api.bulk_df['timestamp'])) 
    utils.live_feed(api=api)                             # now download data live 
    
    #print(len(api.data_df))
    #utils.download_last_n_days(api=api,n=1,path=data_fp)
    #print(len(api.data_df))
    
    
    
    
    
    
if __name__=='__main__':
    get_data()