
import pandas as pd 

import ta.momentum as tam 
import ta.volume as tav 
import ta.volatility as tavl
import ta.trend as tat 
import ta.others as tao

from utils import Utils
import numpy as np  
import pandas as pd 
import random
import torch 
from scipy.stats import norm
import matplotlib.pyplot as plt 


class indicators:
    def __init__(self,df) -> None:
        self.df=df
        self.nlq=self.make_nlq() # non lineear quantiles 
        self.funs_d={
            'rsi':self.fun_rsi
        }

        
        
        
    def make_nlq(self,N=50, plot_res=False ):  # returns list of values betweeb 0-1 inclusive with non linear distribution 
        x=[i/N for i in range(N+1)] # linear percentiles 
        f= lambda x: np.round(1/(1 + np.exp(-(x-0.5 )*10 )),3) # sigmoid function is the best function out there anon 
        yy=[f(i) for i in x]
        yy=(yy-min(yy))/(max(yy)-min(yy))
        if plot_res:
            plt.plot(x,yy,'-o')
            plt.show()
        return yy
        
    def fun_rsi(self, window: int = 25 , inplace = True ):
        colname = f'f-rsi-close-{window}'
        f= lambda df,window : tam.rsi(close=df['close'],window=window)
        self.df[colname]=f(df=self.df,window=window)
        
        
        
if __name__=='__main__':
    u=Utils()
    df=u.read_csv()
    i=indicators(df)

    for fun in i.funs_d.values():
        fun()
    print(df)