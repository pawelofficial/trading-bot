# this script gets your dataframe of data and feature engineer its to find good entry/exit spots
# points are found based on lookahead and is meant to return dataframe that can be used to train
# something that won't use lookahead 
# points are found in following way
# for each point find its long/short score, where long/short score is ratio of given point  and max/min next N (10)
# points. Once you have those scores you take a top percentile of them to have the best ones 
# next you het lowest/highest of those percentiles since good entry points tend to happen in clusters
# like all entries prior to a pump when youre flat are good 

import pandas as pd 
import sys
import numpy as np 
sys.path.append("..") # makes you able to run this from parent dir 

import myfuns as mf 

from mymath import myMath  

from scipy import stats
##import tabulate 


 # ------------------------------------------------------ funs  
#3. find cool points to long/short
    # returns 1% winner/looser boolean columns (for longs and shorts) and long short scores 
def lookahead_score(row,perc,df,N=25):
    index=int(row['index'])
    tdf=df.iloc[index:index+N]
    close=tdf.iloc[0]['close'] # close is not lookaheady because we know when it's happening sort of 
    long_winner=tdf['high']/close > (1+perc)
    short_winner=tdf['low']/close < (1-perc)
    highs=max(tdf['high'])
    lows=min(tdf['low'])
    longs_score=highs/close # high -> good long entry  - highest candle / your candle 
    shorts_score=close/lows # high => good short entry - your candle / lowest candle 
    return pd.Series([long_winner.any(), short_winner.any(),longs_score,shorts_score] ) 

    # returns percentile value given point falls into, for your iq it will be low 
def get_percentile(row,df,colname):
    p=stats.percentileofscore(df[colname],row[colname])
    return p
    
    # returns df with signals based on percentage gain loss or percentile best longs / shorts 
def example_workflow(df,percentile_score=90,best_of_breed=5):
    # get 1p long short flags and long short scores  
    # PERCENTILE_* are signals based on percentiles
    # lowest_8 are signals based on 
    # best of how many in lowest / highest rolling filter 
    df=df.copy(deep=True)
    df[['PRC_LONG','PRC_SHORT','LONGS_SCORE','SHORTS_SCORE']] =df.apply( lookahead_score, perc=0.0075, df=df,  N=25 ,axis = 1  ) 
    p_longs=df['PRC_LONG']==True     # percentage long  
    p_shorts=df['PRC_SHORT']==True   # percentage short  

    # calculate percentiles for long/short scores 
    df['LONGS_PERCENTILE']=df.apply(get_percentile,df=df,colname='LONGS_SCORE',axis=1)
    percentile_longs=df['LONGS_PERCENTILE']>percentile_score
    df['SHORTS_PERCENTILE']=df.apply(get_percentile,df=df,colname='SHORTS_SCORE',axis=1)
    percentile_shorts=df['SHORTS_PERCENTILE']>percentile_score
    # further filter longs/shorts entries to get rid off clusters 
    
    df['LOWEST_LONGS']=df['low'].rolling(window=best_of_breed,center=True).min()
    lowest_longs=(df['low']==df['LOWEST_LONGS']) & p_longs              # lowest from percentage longs 
    #lowest_longs=(df['low']==df['LOWEST_LONGS']) & percentile_longs     # lowest from percentile longs 
    
    df['HIGHEST_SHORTS']=df['high'].rolling(window=best_of_breed,center=True).max()
    highest_shorts=(df['high']==df['HIGHEST_SHORTS']) & p_shorts          # highest from percentage shorts 
    #highest_shorts=(df['high']==df['HIGHEST_SHORTS']) & percentile_shorts # highest from percentile shorts 
    
    # make bookean columns for vectorbot 
    df['LONGS_SIGNAL']=lowest_longs
    df['SHORTS_SIGNAL']=highest_shorts
    lh_columns=['PRC_LONG', 'PRC_SHORT', 'LONGS_SCORE','SHORTS_SCORE', 'LONGS_PERCENTILE', 'SHORTS_PERCENTILE', 'LOWEST_LONGS','HIGHEST_SHORTS', 'LONGS_SIGNAL', 'SHORTS_SIGNAL']
    return df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts, lh_columns
    
 
# ------------------------------------------------------ stuff 
if __name__=='__main__':
    from utils import Utils as u 
    import matplotlib.pyplot as plt   # imports take long time fren ! 


    df=u().read_csv()
    #2. do some math - aggregate 
    m=myMath()
    df=m.aggregate(df=df,scale=15)
    if 0: # use zigzag 
        df=mf.make_zig_zag(return_df=True)
    df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts, lh_columns = example_workflow(df)
    
    
    # check performance - may take some time 


    # 4. plot and print stuff 
    x1y1= [
        [ df['index'],df['close']],
        [ df['index'][percentile_longs], df['close'][percentile_longs] ],
         [ df['index'][percentile_shorts], df['close'][percentile_shorts] ]

    ]
    x2y2= [ 
        [ df['index'],df['close']],
         [ df['index'][lowest_longs], df['close'][lowest_longs] ],
          [ df['index'][highest_shorts], df['close'][highest_shorts] ]
    
           ]

    print(df[lowest_longs | highest_shorts])
    mf.basic_plot(x1y1=x1y1,x2y2=x2y2,linez=['-xk','^g','vr'])

    #plt.hist(df['SHORTS_SCORE'],bins=25)
    #plt.show()
    