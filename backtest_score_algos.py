# this script takes price action data and computes potential entries based on roi of those points 
# opposite signal can be used as exit signal 

from utils import Utils as u 
import pandas as pd 
import numpy as np 

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def swing_score():
    pass 
    # i need to write an algo that works like a lookahead score but instead of stopping at 
    # N future points it looks at swings ! 
    #
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def lookahead_score(row,perc,df,N=25):
    # this simple algo checks if selling somewhere at future N points will lead to making money
    index=int(row['index'])
    tdf=df.iloc[index:index+N]
    close=tdf.iloc[0]['close'] 
    long_winner=tdf['high']/close > (1+perc)
    short_winner=tdf['low']/close < (1-perc)
    highs=max(tdf['high'])
    lows=min(tdf['low'])
    long_score=highs/close                # roi of a point for long 
    short_score=close/lows                # roi of point for short 
    long_winner_exists=long_winner.any()  # boolean showing if given point does result in roi > perc  for long 
    short_winner_exist=short_winner.any() # as above but for short 
    try:
        long_winner_index=min(tdf[long_winner]['index'])        # first index of corresponding winner 
    except ValueError as er:
        long_winner_index=np.nan
    try:
        short_winner_index=min(tdf[short_winner]['index'])
    except ValueError as er:
        short_winner_index=np.nan
    # this logic removes from winners points whose corresponding winners constitute of one point only
    # one - point winners are problematic if they are due to scam pumps, however catching pico top is also a one point winner
    ###l = len ( long_winner[long_winner==True]  )
    ###if l<2:
    ###    long_winner_exists=False 
    ###l = len ( short_winner[short_winner==True]  )
    ###if l<2:
    ###    short_winner_exist=False 
    return pd.Series([long_winner_exists, short_winner_exist,long_score,short_score,long_winner_index,short_winner_index] ) 
    # cleans up signals 
def clean_lookaheads(df,cols=['LONG_BL','SHORT_BL','LONGS_SCORE','SHORTS_SCORE'],N=10): # removes bl signal from adjacent points 
    long_bl=cols[0]
    short_bl=cols[1]
    long_roi=cols[2]
    short_roi=cols[3]
    zeroed_longs=[]
    zeroes_shorts=[]
    no = N-1
    while no < len(df)-1:
        no+=1
        prev_rows=df.iloc[no-N:no]
        cur_row=df.iloc[no].to_dict()
        a0 = prev_rows[long_bl].any() # any of prev rows have bool 
        a1 = cur_row[long_bl]
        b0 = prev_rows[short_bl].any() # any of prev rows have bool 
        b1 = cur_row[short_bl]
        if a0 and a1 : # and a2: 
            cur_row[long_bl]=0
            df.loc[no,long_bl]=False
            zeroed_longs.append(no)
        if b0 and b1 : # and a2: 
            cur_row[short_bl]=0
            df.loc[no,short_bl]=False   
            zeroes_shorts.append[no]
    return df,zeroed_longs,zeroes_shorts
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def eval_strategy(df,long_entry='LONG_BL',short_entry='SHORT_BL',long_exit='SHORT_BL',short_exit='LONG_BL'): 
    long_portfolio = 100 
    long_amo=0
    short_portfolio = 100 
    short_amo=0
    for no, row in df.iterrows():
        d=row.to_dict()
        
        if d[long_entry] and long_portfolio!=0:     # entering long 
            long_amo = long_portfolio / d['close']            
            long_portfolio = 0 
        if d[long_exit] and long_amo !=0 :          # exiting long 
            long_portfolio = long_amo * d['close']
            long_amo =0 

        
        if d[short_entry] and short_portfolio != 0:          # entering short 
            short_amo = short_portfolio / d['close']         # amo shorted 
            short_portfolio = 0                              # zero short portfolio 
            margin_portfolio = short_amo * d['close']        # money from selling  
                                         
        if d[short_exit] and short_amo != 0:                # exiting short 
            amo = margin_portfolio / d['close']             # amo we can buy now from margin 
            if amo < short_amo:                             # steady lads 
                print('deploying more capital - steady lads')
            short_portfolio = (amo - short_amo) * d['close'] + margin_portfolio #
            margin_portfolio = 0  
            short_amo = 0
            
    if long_amo!=0:                                  # exiting longs 
        long_portfolio=long_portfolio + long_amo*d['close']
        long_amo=0
    if short_amo != 0:                               # exiting shorts 
        short_portfolio=short_portfolio + short_amo*d['close'] + margin_portfolio
        short_amo =0 
        
    return short_portfolio + long_portfolio
        

if __name__=='__main__':
    import datetime 
    df=u().read_csv()           # read csv 

    
    df['index']=df.index        # add index col 
    cols=['LONG_BL','SHORT_BL','LONGS_SCORE','SHORTS_SCORE','LONG_WINNER_INDEX','SHORT_WINNER_INDEX']
    df[cols] =df.apply( lookahead_score, 
                       perc=0.0075, 
                       df=df,
                       N=25 ,
                       axis = 1  ) # run backtest strategy 
    amo=eval_strategy(df)          # evaluate strategy 
    print(amo)                     # check if you made money 
    sers=[df['LONGS_SCORE'],df['SHORTS_SCORE']] 
    u().plot_candles(df=df,sers=sers,longs_bl=df['LONG_BL'],shorts_bl=df['SHORT_BL']) # plot  stuff 
 