from distutils.log import error
from logging import exception
from os import curdir
from requests.api import request
import os 
import regex as re 
import requests
import time
import json, hmac, hashlib, time, requests, base64
from requests.auth import AuthBase
import datetime
import pandas as pd 
import numpy as np 
import pathlib
#from trainingdata import TrainingData
#-- 
class CoinbaseExchangeAuth(AuthBase):
    """
    returns request object with appropriate authorization for coinbase api
    """
    def __init__(self, api_key= None , secret_key= None , passphrase=None ):
        self.this_dir=os.path.dirname(os.path.abspath(__file__))
        if any([api_key is None,secret_key is None,passphrase is None]):
            with open(os.path.join(self.this_dir,'secrets','secrets.json')) as f:
                api_config=json.load(f)
                api_key= api_config['api_key']
                api_secret=api_config['api_secret']
                passphrase=api_config['passphrase']

        self.api_key=api_key
        self.secret_key=api_secret
        self.passphrase=passphrase # this should be put in context variable


    def __call__(self, request):
        timestamp = str(time.time())
        message = timestamp + request.method + request.path_url + (request.body or '')
        hmac_key = base64.b64decode(self.secret_key)
        signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest())
        request.headers.update({
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        })
        return request
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
class coinbase_api():
    """ class for getting stuff from coinbase api """
    def __init__(self,auth):
        self.base_url_ama='https://api.coinbase.com/v2'
        self.base_url_pro='https://api.pro.coinbase.com'
        self.time_format='%Y-%m-%dT%H:%M:%S.%fZ'
        self.auth=auth
        self.d={'epoch':[],'open':[],'close':[],'low':[],'high':[],'volume':[],'timestamp':[]} # dictionary with data 
        self.data_df=pd.DataFrame(self.d)
        self.data_df_len=2000 # num of rows of data_df   # not using 
        self.data_fp=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')  # path to data dir 
        self.bulk_df=pd.DataFrame(columns=self.d.keys())   # df with bulk download data 
        self.asset_id='BTC-USD'
        
    
    
    def reset_data_df(self):
        self.data_df=pd.DataFrame(self.d)

    def send_request(self,url):
        """ returns get request from api """
 #       url='https://api.pro.coinbase.com/products/BTC-USD/candles?&granularity=60&start=2021-09-05T13:01:00.000000Z&end=2021-09-05T13:02:00.000000Z'
        try:
            r = requests.get(url, auth=self.auth)
        except:
            
            print(f"bad request trying again")
            print(url)
            time.sleep(3)
            r = requests.get(url, auth=self.auth)
            if r.status_code !=200:
                raise 
  #      print(f" \n your url: {url}")
        return r

    def build_candle_url(self,start_dt,end_dt,granularity=60):
        """returns url string for  historical data aka candles with specified granularity and time frame"""
        start=start_dt.strftime(format=self.time_format)
        end=end_dt.strftime(format=self.time_format)
        candle_url=self.base_url_pro+f"/products/{self.asset_id}/candles?&granularity={granularity}&start={start}&end={end}"       
        return candle_url

    def get_server_time(self):
        """returns server time and server datetime object truncated to 1 minute """
        r=self.send_request(self.base_url_pro+'/time') # call to api 
        try:
            server_time=json.loads(r.text)['iso'] # server timestamp 
        except: # difficult to catch every api response 
            server_time=(datetime.datetime.now()-datetime.timedelta(hours=2)).isoformat()+'Z' 
            #s=re.sub('(\.','.001',r.text) # sometimes api sends weird response like this {"iso":"2022-05-15T13:06:53Z","epoch":1652620013.} instead of '{"iso":"2022-05-15T13:06:51.778Z","epoch":1652620011.778}' 
            #server_time=json.loads(s)['iso']
            
        server_dt=datetime.datetime.strptime(server_time,self.time_format) # server datetime object
        # truncated to 1 minute 
        server_dt=server_dt-datetime.timedelta(seconds=server_dt.second,microseconds=server_dt.microsecond)
        server_time=server_dt.strftime(format=self.time_format)
        # truncating server_dt according to trunc value 
        return server_time,server_dt
    
    def clock_gen(self,start_dt,end_dt,granularity=60):
        """returns a generator yielding start_dt and later_dt up until end_dt
        later_dt is moved forward in time with respippect to granularity to retrieve max no of data
        which is 300 candles, so for granularity = 60 which is 1m later_dt is shifted 1m*300=5h"""
        while start_dt<end_dt:
            later_dt=start_dt+datetime.timedelta(seconds=granularity*300) # so much math
            yield start_dt,later_dt
            start_dt=later_dt

    def parse_raw_data(self,raw_data,single_candle=False):
        """ parses raw data returned by api into a very nice json """
        parsed_data=json.loads(raw_data)
        date_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        
        if single_candle:
            parsed_data=[parsed_data[-1]] # sometimes api returns more than one candle
        parsed_data.sort(key=lambda x:x[0]) # sorting by epoch
        f=lambda x: [float(i[x]) for i in parsed_data] # api returns list of lists
        d={}
        d['epoch']=f(0) # epoch time 
        d['timestamp']=[datetime.datetime.strptime(datetime.datetime.utcfromtimestamp(i[0]).strftime(self.time_format),date_format) for i in parsed_data]
    
        d['low']=f(1)  # low 
        d['high']=f(2)  # high 
        d['open']=f(3)  # open 
        d['close']=f(4) # close 
        d['volume']=f(5) # volume 
        return d

    def fetch_candles(self,start_dt,end_dt,granularity=60):
        """ returns dictionary with raw data of candles and request metadata """
        metadata={}
        metadata['comment']=""
        metadata['start_dt']=start_dt.strftime(format=self.time_format)
        metadata['end_dt']=end_dt.strftime(format=self.time_format)
        metadata['granularity']=granularity
        metadata['asset_id']=self.asset_id
        candle_url=self.build_candle_url(start_dt,end_dt,granularity)
        metadata['candle_url']=candle_url
        request=self.send_request(candle_url) # would be cool to write to metadata initially failed requests
        metadata['request_status_code']=request.status_code
        raw_data=request.text # getting raw data 
        try:
            if metadata['request_status_code']!=200:
                print(f"request status code {metadata['request_status_code']}")
                parsed_data={}
            else:
                parsed_data=self.parse_raw_data(raw_data)
            metadata['no_of_candles']=len(parsed_data['epoch'])
        except Exception as err:
            print(err)
            print(candle_url)
            print(raw_data)

        return {'request_metadata':metadata,'raw_data':raw_data, 'parsed_data':parsed_data}

    def update_df(self,candle,distinct='epoch'): # inserts candle data to a dataframe in distinct manner or not 
        if distinct not in list(candle['parsed_data'].keys()):
            print('distinct filtering not appplied')
            
        
        if self.data_df.empty: # first row always gets inserted
            self.data_df=pd.concat([self.data_df,pd.DataFrame(candle['parsed_data'])],ignore_index=True)
            return 
        if distinct in list(candle['parsed_data'].keys()):
            if self.data_df[distinct].iloc[-1]==candle['parsed_data'][distinct][0]:
                return # not inserting not distinct stuff 
        
        
        self.data_df=pd.concat([self.data_df,pd.DataFrame(candle['parsed_data'])],ignore_index=True)
        # if len of df reached setting then drop first row first, works only for live feed, not for historical fetch
        if len(self.data_df.index)>=self.data_df_len and False:# and len(candle['parsed_data'][distinct][0])==1:  
            todrop=len(self.data_df.index)-self.data_df_len 
            self.data_df.drop(np.arange(todrop),inplace=True)
            self.data_df.reset_index(drop=True, inplace=True) # drop first row and reset index 
    
        # insert data to df 
        
        
        

    def fetch_last_candle(self,granularity=60):
        """ 
        returns last candle based on granularity and self.asset_id
        """
        # get timestamps 
        _,server_dt=self.get_server_time()
        # truncating start time to start of a timebucket
        start_dt=server_dt-datetime.timedelta(seconds=granularity) # start time in past according to granularity 
        end_dt=server_dt # the end is now 
        # get data from api 
        request_data=self.fetch_candles(start_dt,end_dt,granularity)
        #coinbase api sometimes returns more than one candle if you're unlucky with time of your request
        #this if handles that with single_candle flag
        if request_data['request_metadata']['no_of_candles']!=1:
            request_data['parsed_data']=self.parse_raw_data(request_data['raw_data'],single_candle=True)
            request_data['request_metadata']['comment']=f"request returned {request_data['request_metadata']['no_of_candles']} candles instead of expected 1, truncating parsed values and ,no_of_candles to 1 candle"
            request_data['request_metadata']['no_of_candles']=len(request_data['parsed_data']['epoch'])
        return request_data

    # yields last candle  !! 
    def yield_last_candle(self,granularity=60):
        while True:
            yield self.fetch_last_candle(granularity=granularity)
        


    def fetch_historical_candles(self,start_dt,end_dt,granularity=60):
        """returns generator of historical candles of specific granularity
        granularity= {60, 300, 900, 3600, 21600, 86400} 1m 5m 15m 1h
        note that coinbase api max no of candles is 300 but sometimes it does not return all the data
        """
        gen=self.clock_gen(start_dt,end_dt,granularity=granularity) # get datetimes generator 
 #       while True:
 #           left_dt,right_dt=gen.__next__()
 #           print(left_dt,right_dt)
 #       exit(1)            
        
        previous_left_dt,previous_right_dt=None,None
        while True:
            try:
                left_dt,right_dt=gen.__next__() # getting left and right datetimes 
            except StopIteration:
                #print( f"clock generator finished \n start_dt: {start_dt} \n end_dt: {end_dt} \n left_dt: {previous_left_dt} \n right_dt={previous_right_dt}")
                return
            request_data=self.fetch_candles(left_dt,right_dt,granularity=granularity)
            yield request_data
            previous_left_dt,previous_right_dt=left_dt,right_dt

    def write_dictionary_to_file(self,filename,dic,path="", mode='w',ext='csv',write_header=True,eol='',truncate=True):
        """ writes dic to a flatfile"""
        #write as csv 
        fp=os.path.join(path,filename)
        print(fp)

        if truncate:
            with open(fp,'w') as f:
                print(f"truncating {filename}")
                f.truncate()
        if ext=='csv':
            #print("writing")
            with open (fp,mode) as f:
                keys_list=list(dic.keys())
                if write_header: # writes keys as header
                    print("writing header")
                    f.write(','.join(keys_list))
                    f.write(eol) # not a good idea to use that - use lineterminator parameter in pandas to read csv file like that 
                for no,val in enumerate(dic[keys_list[0]]):
                    line=[str(dic[key][no]) for key in keys_list]
                    f.write('\n')
                    f.write(','.join(line)+eol)
                    
        if ext=='json':
            raise "dupa, not implemented"
        #      with open(path+filename+'.json','w') as f:
#            json.dump(request_data,f,indent=4)




    def bulk_download_data(self
                           ,start_dt='01-01-2023'
                           ,end_dt='01-02-2023'
                           ,granularity=60
                           ,to_df=False,to_csv=True,fp=None,fname=None):
        """downloads data from start_dt to end_dt into a csv or a df"""
        start_dt=datetime.datetime.strptime(start_dt, '%d-%m-%Y')
        end_dt=datetime.datetime.strptime(end_dt, '%d-%m-%Y')
        header=True   # first write to file with header 
        mode='w'      # first write to file with write mode  
        if fp is None: 
            start_date=start_dt.strftime('%Y%m%d_%H%M%S')
            end_date=end_dt.strftime('%Y%m%d_%H%M%S')
            if fname is None:
                fname=f'bulk_download_{start_date}_{end_date}_{self.asset_id}_{granularity}.csv'
            fp=os.path.join(self.data_fp,fname)
            
        gen=self.fetch_historical_candles(
            start_dt=start_dt,
            end_dt=end_dt,
            granularity=granularity)
        
        while True:
            
            print('downloading ! ')
            try:
                request_data=gen.__next__()
            except StopIteration as err:
                print("end of iteration ")
                break 
        
            tmp_df=pd.DataFrame.from_dict(request_data['parsed_data'])
            if to_df:
                self.bulk_df = pd.concat([self.bulk_df, tmp_df], ignore_index=True)
            
            if to_csv:
                tmp_df.to_csv(fp,mode=mode,header=header,index=False,sep='|')
                header=False
                mode='a'
        
        
        return self.bulk_df,fp


class ApiUtils:
    def __init__(self,CoinbaseExchangeAuth,coinbase_api) -> None:
        self.CoinbaseExchangeAuth=CoinbaseExchangeAuth
        self.coinbase_api=coinbase_api
        pass
    
    def init(self,asset_id: str = 'BTC-USD',config_filepath='.\\credentials\\api_config.json'): # setup coinbase api instance
        print(os.getcwd())
        api_config=json.load(open(config_filepath))
        auth=CoinbaseExchangeAuth(
            api_config['api_key'],
            api_config['api_secret'],
            api_config['passphrase'])
        api=coinbase_api(auth)
        api.asset_id=asset_id
        return api
    
    def download_last_n_days(self,api,n=7,path='./'): # downloads data for last n days 
        _,server_dt=api.get_server_time()
        start_dt=server_dt-datetime.timedelta(days=n)
        api.bulk_download_data(
        start_dt=start_dt,
        end_dt=server_dt,
        path=path, # os.path.abspath(os.getcwd())+"\\ETL\\rawdata\\",
        mode='a',
        ext='csv',
        write_header=True,
        truncate=True,
        granularity=60,
        filename=f"{api.asset_id}{start_dt.isoformat()[:10]}_{server_dt.isoformat()[:10]}"+'.csv' )
        return f"{api.asset_id}{start_dt.isoformat()[:10]}_{server_dt.isoformat()[:10]}"+'.csv'
        
    def download_by_dates(self,api,start_date='01-01-2022',end_date='30-01-2022',path='./'):
        start_dt=datetime.datetime.strptime(start_date, '%d-%m-%Y')
        end_dt=datetime.datetime.strptime(end_date, '%d-%m-%Y')
        api.bulk_download_data(
        start_dt=start_dt,
        end_dt=end_dt,
        path=path, # os.path.abspath(os.getcwd())+"\\ETL\\rawdata\\",
        mode='a',
        ext='csv',
        write_header=True,
        truncate=True,
        granularity=60,
        filename=f"{api.asset_id}{start_dt.isoformat()[:10]}_{end_dt.isoformat()[:10]}"+'.csv' )
        return f"{api.asset_id}{start_dt.isoformat()[:10]}_{end_dt.isoformat()[:10]}"+'.csv' 

        
    def live_feed(self,api,fakedf=None): # loop for live feeding data on cdistinct epoch 
    # if fakedf is given life feed uses this df to return data rather than data from api
        if fakedf is not None:
            i=0
            while i<len(fakedf.index):
                row=fakedf.iloc[i]
                row['msg']='fake live feed'
                api.data_df.loc[len(api.data_df.index)]=row
                print(api.data_df)
                i+=1
                time.sleep(1)
            return 
        
        while True:
            candle=api.fetch_last_candle()
            candle['parsed_data']['msg']=time.strftime('%X')
            api.update_df(candle=candle,distinct='epoch') # distinct epoch filtering 
            print('------',datetime.datetime.now(),'------',len(api.bulk_df))
            print(api.data_df)
            
            time.sleep(1)
            
    
    def feed_df(self,api,n=1,m=0): # feeds dataframe with historical data 
        _,server_dt=api.get_server_time()
        start_dt=server_dt-datetime.timedelta(days=n)
        end_dt=server_dt-datetime.timedelta(days=m)
        print(start_dt,end_dt)
        if end_dt<=start_dt:
            print('wrong time window')
            return
        gen=api.fetch_historical_candles(
                start_dt=start_dt,
                end_dt=end_dt,
                granularity=60)
        while True:
            try:
                request_data=gen.__next__()
            except StopIteration as err:
                print("end of iteration ")
                break 
            api.update_df(candle=request_data)
        return api.data_df


if __name__=='__main__':
    auth=CoinbaseExchangeAuth()
    ca=coinbase_api(auth)
    ca.bulk_download_data()

if __name__=='__main__x':
    fp= os.path.abspath('./secrets/secrets.json')
    out_fp=os.path.abspath('./data/')

    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)

    api=utils.init(config_filepath=fp)
    if 0:  # downloads  last n days 
        utils.download_last_n_days(api=api,n=2,path=out_fp)
    if 1: 
        utils.download_by_dates(api=api,start_date='01-01-2022',end_date='03-01-2022',path=out_fp)
        
    if 0:# updates api.data_df with historical data 
        histdf=utils.feed_df(api=api,n=2,m=1)
    if 0:  # loops download in a while loop to a api.data_df
        fake_df=histdf
        api.reset_data_df()
        utils.live_feed(api=api,fakedf=fake_df)
    