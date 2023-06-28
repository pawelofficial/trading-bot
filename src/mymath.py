import sys
sys.path.append("..") # makes you able to run this from parent dir 
import string

import pandas as pd 
import numpy as np 
import random 

import myfuns as mf
import datetime
from scipy import integrate

import datetime
class myMath:
    def __init__(self) -> None:
        self.tformat='%Y-%m-%dT%H:%M:%S.%fZ'
        self.tformat='%Y-%m-%d %H:%M:%S'
        self.lambda_d = {
            'ema': lambda df,col,window : df[col].ewm(span=window).mean()
            ,'sma': lambda df,col,window : df[col].rolling(window=window).mean()
            ,'std': lambda df,col,window: df[col].rolling(window=window).std()
            ,'env-dist':lambda df,x,y,z,window : ( df[x] -  (df[y] + df[z] ) / 2 )  /   (df[y] - df[z] ) *2 # envelope distance fren , what did you expect from this comment ? 
            ,'grad' : lambda df,col,window : np.gradient(df[col],window) # 1d gradient because gradient is important ! 
            ,'cumdiff': lambda df,col1,col2,window: df['index'].rolling(window=window).apply(lambda x: self.cumdiff(df,x,col1,col2)) # basically any  function on rolling window
            

   
            ,'floor_dt': lambda df,str_col, scale: df[str_col].apply(lambda x: datetime.datetime.strptime(x,self.tformat) -
                    datetime.timedelta(
                    minutes=( datetime.datetime.strptime(x,self.tformat).minute) % scale,
                    seconds= datetime.datetime.strptime(x,self.tformat).second,
                    microseconds= datetime.datetime.strptime(x,self.tformat).microsecond)
                                                                     )
            , 'ts': lambda df,str_col:  df[str_col].apply(lambda x: datetime.datetime.strptime(x,self.tformat ) )  # converts str col to timestamp 
            ,'hod': lambda df,str_col:  df[str_col].apply(lambda x: datetime.datetime.strptime(x,self.tformat).hour ) # hour of a day 
            ,'moh': lambda df,str_col:  df[str_col].apply(lambda x: datetime.datetime.strptime(x,self.tformat).minute )  # hour of a day 
            ,'dow': lambda df,str_col:  df[str_col].apply(lambda x: datetime.datetime.strptime(x,self.tformat).weekday() )  # hour of a day 
            ,'dom': lambda df,str_col:  df[str_col].apply(lambda x: datetime.datetime.strptime(x,self.tformat).day )  # hour of a day 
            
        } 
        # dictionary for functions for aggregation - aka for close you want last, for high you want max ! 
        self.agg_funs_d={  'open':lambda ser: ser.iloc[0],
                 'close':lambda ser: ser.iloc[-1],
                 'high':lambda ser: ser.max(),
                 'low': lambda ser: ser.min(),
                 'volume': lambda ser: ser.mean()
                }  
        
        self.math_df=pd.DataFrame({})
    def cumdiff(self,df,x,col1,col2):
        c= df.iloc[x][col1]-df.iloc[x][col2] # difference between two cols 
#        i=integrate.cumtrapz(c) # not sure why not just sum this shiet anon, since youre using index, but cum integral is more advanced than suming ! 
        return np.sum(c)
    # reads csv into math_df 
    def read_csv(self,filepath : str, inplace : bool = True  ) ->pd.DataFrame:
        if inplace:
            self.math_df=pd.read_csv(filepath)
            return pd.DataFrame({})
        return pd.read_csv(filepath,lineterminator=';')    
    # calculates stuff  from lambda_ d and puts it into df 
    def calculate_fun(self,df: pd.DataFrame, fun_name : str, newcolname : str = '',  inplace : bool = True, **kwargs ) -> pd.Series:
        if df.empty:
            df=self.math_df
        fun = self.lambda_d[fun_name] # anon function no cap
         
        if newcolname=='':  # auto name new column if newcolname was not specified 
            newcolname = fun_name + '-'+ '-'.join ( [ str(i) for i in  kwargs.values() ] )
        fun_args = fun.__code__.co_varnames[1:] # lambda function arguments, not used herebut can be used for error handling 
        ser=  fun(df, *list(kwargs.values( ))  ) 
        if inplace:
            df[newcolname]= ser
        return ser 
    # calculates rolling nadaraya 
    def rolling_nadaraya(self,df,xcol,ycol,window=1000,h=1.0) -> pd.Series: #basically rolling nadaraya, should put that to to rolling 
        def nadaraya(Xi,Yi,x,h=1.0):
            norm_z = lambda x,mu,sigma : 1/(sigma * np.sqrt (2* np. pi )) * np.exp ( -1/2 * ( (x-mu)/sigma  )**2  )
            norm= lambda x,mu : norm_z(x,mu,1) # norm od liczby 

            w=np.zeros(len(Xi))
            y=np.zeros(len(Xi))
            foo= np.sum ( [ norm(x,xj ) for xj in Xi    ] )
            for i in range(len(Xi)):
                w[i]=norm(x/h,Xi[i]) / foo 
                y[i] = w[i]*Yi[i] /h 

            return np.sum(y)  

        y_nad=[]
        for index,row in df.iterrows():
            start=index-window if index>window else 0        
            Xi=list(df.iloc[start:index+1][xcol] )
            Yi=list(df.iloc[start:index+1][ycol] ) 
            x=float(row[xcol])
            y_nad.append(nadaraya(Xi=Xi,Yi=Yi,x=x,h=h))
        return y_nad
    # returns a daraframe aggregated to a timeframe using a src column name 
    def aggregate(self,df :pd.DataFrame = pd.DataFrame({}), scale : int = 5, src_col : str = 'timestamp',cols : list = ['open','close','low','high','volume']):
        if df.empty:
            df=self.math_df
        if src_col not in df.columns:
            print('src col not in df columns')
            raise 
        dt_col = '-'.join(['ts',str(scale) ])
        agg_df=pd.DataFrame({})
        
        df[dt_col]=self.calculate_fun(df=df,fun_name='floor_dt',str_col='timestamp',scale=scale) # need to write floor column to df  to later groupby it 
        agg_df[dt_col]=df[dt_col].unique().copy()
        for col in cols:
            g=df[[col,dt_col]].groupby([dt_col])
            ser=g.apply(self.agg_funs_d[col])[col].reset_index(name=col)
            agg_df=agg_df.merge(ser,left_on=dt_col,right_on=dt_col)

        #agg_df['index']=agg_df.index
        return agg_df
    # dumps df to csv      
    def dump_csv(self,df :pd.DataFrame(),filename : str ='math_df' ,cols :list =[] ):
        if cols ==[]:
            cols=df.columns
        filename=filename.replace('.csv','')+'.csv'
        df[cols].to_csv(filename,columns=cols,index=False,quotechar='"',na_rep='',quoting=1)
        
# this class appends strategy-entry  and strategy-exit boolean columns to math_df  as well as other boolean columns, columns are in capital to know they come from here 
class myEntries(myMath):
    def __init__(self,math_df : pd.DataFrame({}) ):
        super().__init__()
        self.df=math_df 
    def check_keys(self, keys : list = []): # checks if passed list of keys exist in self.df 
        for i in keys:
            if i not in self.df.columns:
                print(f'item {i} is not in df columns')
                print(self.df)
                raise 
        
    def nadaraya_watson_strategy(self,col : str ='close', inplace=True):  # checks if ndr+05 and ndr-05 columns were pierced by a close 
        self.check_keys(keys=['ndr-05','ndr+05',col])
        if inplace:
            self.df['NW-EXIT']=self.df[col]>self.df['ndr+05']
            self.df['NW-ENTRY'] = self.df[col]<self.df['ndr-05']
            return 
        return pd.DataFrame({'NW-EXIT':self.df[col]>self.df['ndr+05'], 'NW-ENTRY':self.df[col]<self.df['ndr-05'] } ) 

    def ema_grad(self,col : str = 'ema', inplace : bool = True ): # adds ema differential column ema-grad and ema-grad-sign boolean column to knowhwen it's positive 
        self.check_keys(keys=['ema-grad'])
        if inplace:
            self.df['EMA-GRAD-SIGN']=self.df['ema-grad']>=0
        
        



def make_math_df(df :pd.DataFrame, window : int = 25 ):
    import matplotlib.pyplot as plt 
    import pandas_ta as ta
    m=myMath()
    df['index']=df.index
    df['ema']=m.calculate_fun(df=df,fun_name='ema',col='close',window=window,inplace=False)
    df['ema-grad']=m.calculate_fun(df=df,fun_name='grad',col='ema',window=window,inplace=False )
    df['ndr']=m.rolling_nadaraya(df=df,xcol='index',ycol='close',h=1)
    df['ndr+05']=df['ndr'] + m.calculate_fun(df=df,fun_name='std',col='close',window=20,inplace=False)*0.5
    df['ndr-05']=df['ndr'] + m.calculate_fun(df=df,fun_name='std',col='close',window=20,inplace=False)*0.5*-1
    df['ndr+075']=df['ndr'] + m.calculate_fun(df=df,fun_name='std',col='close',window=20,inplace=False)*0.75
    df['ndr-075']=df['ndr'] + m.calculate_fun(df=df,fun_name='std',col='close',window=20,inplace=False)*0.75*-1
    df['ndr+1']=df['ndr'] + m.calculate_fun(df=df,fun_name='std',col='close',window=20,inplace=False)*1
    df['ndr-1']=df['ndr'] + m.calculate_fun(df=df,fun_name='std',col='close',window=20,inplace=False)*1*-1
    # important columns:
    df['env-dist']= m.calculate_fun(df=df,fun_name='env-dist',x='close', y= 'ndr+05',z='ndr-05'  , window=20,inplace=False) 
    df['ndr-grad']= m.calculate_fun(df=df,fun_name='grad', col='ndr',window=10) 
    df['rsi']=mf.calc_rsi(df['close'], lambda s: s.ewm(span=25).mean())
    df['rsi-ema']=m.calculate_fun(df=df,fun_name='ema', col='rsi',window=25,inplace=False) 
    df['rsi-ema-grad']= m.calculate_fun(df=df,fun_name='grad', col='rsi-ema',window=25,inplace=False) 
    df['ema-area']= m.calculate_fun(df=df,fun_name='cumdiff',x='close', y= 'ema',window=25,inplace=False) 
    stoch = df.ta(kind='stoch',high='high', low='low', k=14, d=3, append=False)
    df['stoch']=stoch['STOCHk_14_3_3']>stoch['STOCHd_14_3_3']
    return df 


# using existing math df and adding strategy to it 
if __name__=='__main__':
    m=myMath()
    filepath='math_df_5min_BTC-USD2022-06-04_2022-06-05.csv'
    filepath='math_df_60min_BTC-USD2022-03-07_2022-06-05.csv'
    df=mf.read_csv(filename=filepath)
    s=myEntries(math_df=df)
    s.nadaraya_watson_strategy()
    s.ema_grad()
    df=s.df
    print(df)
    
    nw_entry_mask=df['NW-ENTRY']
    nw_exit_mask=df['NW-EXIT']

    x1y1=[
        [df['index'],df['close']], 
        [df['index'],df['ndr-05']], 
        [df['index'],df['ndr+05']]
        
        , [df['index'][nw_entry_mask] ,df['close'][nw_entry_mask] ]      
        , [df['index'][nw_exit_mask] ,df['close'][nw_exit_mask] ]      
        
        ]
    x2y2=[
        [df['index'],df['ema-grad']],
        [df['index'],df['rsi-ema-grad']],
         [df['index'],df['ema-area']]
        
    ]
    linez=['-k','-g','-r','^b','vr']
    mf.basic_plot(x1y1=x1y1,x2y2=x2y2,linez=linez)
    

# making a math df 
if __name__ =='__main__X':
    nw=datetime.datetime.now()
    hl=mf.how_long()
    m=myMath()
    filepath='BTC-USD2022-03-07_2022-06-05'
#    filepath='BTC-USD2022-06-04_2022-06-05'

    m.read_csv(filepath = f'./{filepath}.csv')
    scale=60
    m.math_df=m.aggregate(df=m.math_df, scale =scale, src_col='timestamp')
    df=make_math_df(m.math_df.copy(deep=True))
    df.rename({'ts-5':'timestamp','rsi-ema':'rsi'})#,'STOCHk_14_3_3':'stoch1','STOCHd_14_3_3':'stoch2'})
    cols_to_dump=['index','open','close','low','high','ema','env-dist','rsi-ema','rsi-ema-grad','ema-area','stoch','ndr+05','ndr-05','ema-grad']
    
    m.dump_csv(df=df,filename=f'math_df_{scale}min_{filepath}',cols=cols_to_dump)

    print(df)


    mf.basic_plot(x1y1 = [
                         [ df['index'],df['close'] ]
                         ,[ df['index'],df['ndr+05'] ]
                          ,[ df['index'],df['ndr-05'] ]
#                         , [ df['index'],df['ndr+05'] ]
#                         , [ df['index'],df['ndr-05'] ]
                        ],    
                 x2y2 = [ 
                         [df['index'],df['ema-area'] ]
                         ,[df['index'],df['ema-area'] ]
#                         [ df['index'],df['rsi-ema'] ]
#                        , [ df['index'],df['rsi-ema-grad'] ]
                        ]
                 ,show=1
                 )

    


