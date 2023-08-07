import unittest
import sys
import os
import logging


logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)) , './logs/coinbase_api_tests.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)

# do 
# cd C:\gh\trading-bot\_src
# python -m unittest bot.tests.tests_coinbase_api

if __name__!='__main__':
    from bot.coinbase_api import *
else:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from coinbase_api import *


os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class test_coinbase_api(unittest.TestCase):
    def setUp(self):
        self.this_dir=os.path.dirname(os.path.abspath(__file__))
        self.parent_dir=os.path.dirname(self.this_dir)
        self.secrets_fp=os.path.join(self.parent_dir,'secrets','secrets.json')
        
        with open(self.secrets_fp) as f:
            api_config=json.load(f)
        self.auth=CoinbaseExchangeAuth(
            api_config['api_key'],
            api_config['api_secret'],
            api_config['passphrase'])
        self.ca=coinbase_api(self.auth)
        self.ca.asset_id='BTC-USD'

        
    def test__build_candle_url(self,start_dt=None,end_dt=None,granularity=60*60):
        if start_dt is None:
            start_dt=datetime.datetime.strptime('01-01-2023', '%d-%m-%Y')
        if end_dt is None:
            end_dt=datetime.datetime.strptime('02-01-2023', '%d-%m-%Y')
        url=self.ca.build_candle_url(start_dt,end_dt,granularity=granularity)
        logging.info(f'test__build_candle_url url: {url} ')
        return url 
    
    def test__send_request(self,url=None):
        if url is None:
            url=self.test__build_candle_url()
        r=self.ca.send_request(url)
        logging.info(f'test__send_request r.text: {r.text} ')
        logging.info(f'test__send_request r.status_code: {r.status_code} ')
        return r
        
    def test__parse_raw_data(self,raw_data=None):
        if raw_data is None:
            raw_data=self.test__send_request().text
        parsed_data=self.ca.parse_raw_data(raw_data)
        logging.info(f'test__parse_raw_data parsed_data: {parsed_data} ')
        
    def test__fetch_last_candle(self,granularity=60):
        last_candle=self.ca.fetch_last_candle(granularity=granularity)
        logging.info(f'test__fetch_last_candle last_candle: {last_candle} ')
        
    def test__yield_candles(self,granularity=60):
        gen=self.ca.yield_last_candle(granularity=granularity)
        j=0
        while  j<10:
            j+=1
            logging.info(f'test__yield_candles gen: {next(gen)} ')
            time.sleep(1)
        
        
    def test__bulk_download_data(self):
        start_dt='01-01-2023' 
        end_dt='01-02-2023'   
        granularity=60
        bulk_df=self.ca.bulk_download_data(start_dt,end_dt,granularity=granularity)
        logging.info(f'test__bulk_download_data bulk_df')


  



if __name__ == '__main__':
    #unittest.main()
    t=test_coinbase_api()
    t.setUp()
    t.test__yield_candles()