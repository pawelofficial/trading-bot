import numpy as np 
import pandas as pd 
import random 
import matplotlib.pyplot as plt 
from typing import Callable
import datetime
import scipy 
from scipy import stats

def test_population(N: int = 100 ,uniform_x : int =True, zeros_and_ones : bool = False ):
    
    noise=50
    f= lambda x: x*x*np.cos(x)
    if uniform_x:
        xi=np.linspace(-10,10,N)
    else:
        xi=[random.randint(-1000,1000)/100 for i in range(N)] # less likely to have duplicate sis 
    eps=[1*random.randint(0,noise) for i in range(N)]
    yi=[f(xi[i])+eps[i] for i in range(N)]
    if zeros_and_ones:
        xi=np.zeros(len(xi))
        yi=np.ones(len(yi))
    return pd.DataFrame({'Xi':xi,'Yi':yi})
    
# plots stuff for fast plotting 

def basic_plot(x1y1,x2y2 : list = [], normalize : list = [0,0] , linez : list = [], show : bool  = True ):# plots x1y1 and x2y2 where xiyi is  -> [ [xi,yi],[xj,yj],[xk,y,],]
    if not show:
        return
    
    if type(x1y1[0])!=type([]):
        raise # x1y1 has to be [ [x1,y1],[x11,y11],[x111,y111] ...   ] -> list of x1,y1 pairs 
    normalize_f = lambda x: 1 
    if normalize : 
        normalize_f = lambda x: np.max(x)
    
    if linez==[]:
        linez=['.-' for i in range(len(x1y1))  ]
        linez[0]='-o'
    
    fig,ax=plt.subplots(2,1)
    ax[1].get_shared_x_axes().join(ax[0],ax[1] )
    mintick=99999999999
    maxtick=-mintick
    for i in range(0,len(x1y1)):
        x2=x1y1[i][0]
        y2=x1y1[i][1]
        if normalize[0]:   
            y2 = y2 /  np.max(y2)
        if len(y2)==0:
            print('cant plot empty sequence ')
            return
        mintick=min( [mintick, min(y2)]) 
        maxtick = max([maxtick,max(y2)])
        ax[0].plot(x2,y2,linez[i], label = y2.name, linewidth=1)
   # pot x2ye data 
    major_ticks1=np.arange(mintick,maxtick,(maxtick-mintick)/20)
 
 
    mintick=99999999999
    maxtick=-mintick
    for i in range(0,len(x2y2)):
        x2=x2y2[i][0]
        y2=x2y2[i][1] 
        if normalize[1]:
            y2 = y2 / np.max(y2)
        mintick=min( [mintick, min(y2)]) 
        maxtick = max([maxtick,max(y2)])
        ax[1].plot(x2,y2,linez[i], label = y2.name)
    major_ticks2=np.arange(mintick,maxtick,(maxtick-mintick)/20)
        

   # sow plot 
    ax[0].grid(True)
    ax[1].grid(True)
    if normalize[0]:
        ax[0].set_ylim(-1,1)
    else:
        ax[0].set_yticks(major_ticks1)
    if normalize[1]:
        ax[1].set_ylim(-1,1)
    else:
        ax[1].set_yticks(major_ticks2)
    
    ax[0].legend()
    ax[1].legend()

    plt.show()

def calc_rsi(over: pd.Series, fn_roll: Callable) -> pd.Series: # this code is totally not from stack overflow ! 
    # Get the difference in price from previous step
    delta = over.diff()
    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:] 

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()

    roll_up, roll_down = fn_roll(up), fn_roll(down)
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Avoid division-by-zero if `roll_down` is zero
    # This prevents inf and/or nan values.
    rsi[:] = np.select([roll_down == 0, roll_up == 0, True], [100, 0, rsi])
    rsi.name = 'rsi'
    return rsi



def make_syntetic_signal_df(no_of_rows : int = 100 ):
    random.seed(10)
    
    N=int(no_of_rows*0.25) # no of neither true or false 
    pa=[random.randint(90,110)/100 for i in range(no_of_rows)]
    signals=[ int(i<1) for i in pa ] # if pa < 1 buy, if pa>1 sell 
    for i in range(N): # make some signals neither True or False 
        index=random.randint(0,len(pa)-1)
        signals[index]=-1
    df=pd.DataFrame({'close':pa,'signal':signals})
    df.iloc[-1]=df.iloc[0] # making last row same as first row to see net worth better 
    return df 

# makes a zig zagi zigzag 
def make_zig_zag(N : int = 10,mu=1,n=4,minmax=0.1,return_df : bool = True ):
    n=n*2
    x=list(np.linspace(0,1,N*n))
    y=list(np.linspace(mu-minmax,mu+minmax,N))
    Y=y.copy()
    for i in range(n-1):
        if i % 2 ==0:
            Y=Y+sorted(y,reverse=True)
        else:
            Y=Y+y
            
    if return_df:
        signal=[]
        for i in Y:
            if i<mu-minmax/2:
                signal.append(-1)
            elif i>mu+minmax/2:
                signal.append(1)
            else:
                signal.append(0)
        df=pd.DataFrame({'close':Y,'signal':signal})
        df['index']=df.index
        return df 
    return x,Y



def read_csv(filename,lineterminator=';',add_index : bool = False, cols : list =[] ):
    if add_index:
        df=pd.read_csv(filename,lineterminator=lineterminator)
        df['index']=df.index
        if len(cols)!=1:
            return df[cols].copy(deep=True)
        return df  
    
    return pd.read_csv(filename,lineterminator=lineterminator)

def dump_csv(df,filename):
    df.to_csv(filename,index=False)

def how_long():
    """how_long=f.how_long() # do this in your script to know how long things took 
        print(how_long())"""    
    nw=datetime.datetime.now()
    how_long = lambda : (datetime.datetime.now() - nw ).total_seconds()
    return how_long


# this function returns timeframes sliced by a function provided so it's not uniform but can be non uniform 
# it's not perfect 

def get_timeframes(dist_fun: str = 'exp', timeframes_ranges : list = [3,200], N : int = 10):
    # imagine you want to divide a range but not in a linear fashion - this function does that 
    # you can marvel at it's beauty, starting .... now : 
    N=N-1
    dist_funs={'exp':lambda x: np.exp(x),
           'linear': lambda x: x ,
           'power': lambda x : x**3.5 }
    if dist_fun not in dist_funs.keys():
        print('dupa ! ')
        raise 
    dist_fun=dist_funs[dist_fun]
    # find max value that fits the timeframes ranges 
    x=0
    while dist_fun(x)<=timeframes_ranges[1]:
        x=x+1
    x=x-1 
    # find min value that fits the timeframes ranges 
    i=0 
    while dist_fun(i)<=timeframes_ranges[0]:
        i=i+1 
    
#    print(dist_fun(i),i)
#    print(dist_fun(x),x)
    X=np.append(np.arange(i,x,(x-i)/N ),x) # argument for dist fun is spaced linearly 
    XX=[int(dist_fun(x)//1) for x in X]            # dist_fun values for argument that fit between timeframe_ranges 
    return XX 
    
    


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

def get_percentile(row,df,colname):
    p=stats.percentileofscore(df[colname],row[colname])
    return p

def example_workflow(df,percentile_score=80):
    # get 1p long short flags and long short scores  
    df[['1p_long','1p_short','longs_score','shorts_score']] =df.apply( lookahead_score, perc=0.015, df=df,  N=10 ,axis = 1  ) 
    p_longs=df['1p_long']==True
    p_shorts=df['1p_short']==True 

    # calculate percentiles for long/short scores 
    df['longs_percentile']=df.apply(get_percentile,df=df,colname='longs_score',axis=1)
    percentile_longs=df['longs_percentile']>percentile_score
    df['shorts_percentile']=df.apply(get_percentile,df=df,colname='shorts_score',axis=1)
    percentile_shorts=df['shorts_percentile']>percentile_score
    # further filter longs/shorts entries to get rid off clusters 
    df['lowest_longs']=df['close'].rolling(window=5,center=True).min()
    lowest_longs=(df['close']==df['lowest_longs']) & p_longs
    df['highest_shorts']=df['close'].rolling(window=5,center=True).max()
    highest_shorts=(df['close']==df['highest_shorts']) & p_shorts

    # make bookean columns for vectorbot 
    df['LONG_ENTRY']=df['close']==df['lowest_longs']
    df['LONG_EXIT']=df['close']==df['highest_shorts']
    return df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts


def plot_candlestick2(candles_df : pd.DataFrame
                      ,x1y1 : list =[]
                      , x2y2 : list = []
                      ,longs_ser : pd.Series = pd.Series({},dtype=np.float64)
                      ,shorts_ser : pd.Series = pd.Series({},dtype=np.float64)
                      ):
    df=candles_df
    plt.rcParams['axes.facecolor'] = 'y'
    low=df['low']
    high=df['high']
    open=df['open']
    close=df['close']
    # mask for candles 
    green_mask=df['close']>=df['open']
    red_mask=df['open']>df['close']
    up=df[green_mask]
    down=df[red_mask]
    # colors
    col1='green'
    black='black'
    col2='red'

    width = .4
    width2 = .05

    fig,ax=plt.subplots(2,1)
    ax[0].bar(up.index,up['high']-up['close'],width2,bottom=up['close'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['low']-up['open'],width2, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['close']-up['open'],width, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(down.index,down['high']- down['close'],width2,bottom=down['close'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['low']-  down['open'],width2,bottom=down['open'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['close']-down['open'],width,bottom=down['open'],color=col2,edgecolor=black)

    for xy in x1y1:
        ax[0].plot(xy[0],xy[1])
        
        
    for xy in x2y2:
        ax[1].plot(xy[0],xy[1])

    if not longs_ser.empty:
        msk=longs_ser==True
        ax[0].plot(longs_ser[msk].index, df[msk]['low']*longs_ser[msk].astype(int),'^g')

    if not shorts_ser.empty:
        msk=shorts_ser==True
        ax[0].plot(shorts_ser[msk].index, df[msk]['high']*shorts_ser[msk].astype(int),'vr')


# plts candlestick fren 
def plot_candlestick(df
                     , df2 :pd.DataFrame = pd.DataFrame({})
                     , shorts_ser:pd.Series = pd.Series(dtype=float)
                     , longs_ser:pd.Series = pd.Series(dtype=float)
                     , real_longs:pd.Series = pd.Series(dtype=float)
                     , real_shorts:pd.Series = pd.Series(dtype=float)
                     ):
    plt.rcParams['axes.facecolor'] = 'y'
    low=df['low']
    high=df['high']
    open=df['open']
    close=df['close']
    # mask for candles 
    green_mask=df['close']>=df['open']
    red_mask=df['open']>df['close']
    up=df[green_mask]
    down=df[red_mask]
    # colors
    col1='green'
    black='black'
    col2='red'

    width = .4
    width2 = .05

    fig,ax=plt.subplots(2,1)
    ax[0].bar(up.index,up['high']-up['close'],width2,bottom=up['close'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['low']-up['open'],width2, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['close']-up['open'],width, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(down.index,down['high']- down['close'],width2,bottom=down['close'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['low']-  down['open'],width2,bottom=down['open'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['close']-down['open'],width,bottom=down['open'],color=col2,edgecolor=black)
        
    if 'LONGS_SIGNAL' in df.columns:
        msk=df['LONGS_SIGNAL']==1
        ax[0].plot(df[msk].index, df[msk]['low']*df[msk]['LONGS_SIGNAL'],'^g')
    if 'SHORTS_SIGNAL' in df.columns:
        msk=df['SHORTS_SIGNAL']==1
        ax[0].plot(df[msk].index, df[msk]['high']*df[msk]['SHORTS_SIGNAL'],'vr')
        
        
        
    if not shorts_ser.empty:
        mask=shorts_ser>0
        ax[0].plot(shorts_ser.index[mask],shorts_ser[mask],'vr')

    if not longs_ser.empty:
        mask=longs_ser>0
        ax[0].plot(longs_ser.index[mask],longs_ser[mask],'^g')
        
    if not real_longs.empty:
        ax[0].plot(real_longs.index,real_longs,'og')

    if not real_shorts.empty:
        ax[0].plot(real_shorts.index,real_shorts,'or')
        
    if not df2.empty:
        ax[1].plot(df2.index,df2,'x')
    return ax 
    
def aggregate(df :pd.DataFrame = pd.DataFrame({}), 
              scale : int = 5, 
              src_col : str = 'timestamp',
              cols : list = ['open','close','low','high']):
    tformat='%Y-%m-%dT%H:%M:%S.%fZ'
    floor_dt =  lambda df,str_col, scale: df[str_col].apply(lambda x: datetime.datetime.strptime(x,tformat) -
                    datetime.timedelta(
                    minutes=( datetime.datetime.strptime(x,tformat).minute) % scale,
                    seconds= datetime.datetime.strptime(x,tformat).second,
                    microseconds= datetime.datetime.strptime(x,tformat).microsecond))
    
    
    agg_funs_d={  'open':lambda ser: ser.iloc[0],
                 'close':lambda ser: ser.iloc[-1],
                 'high':lambda ser: ser.max(),
                 'low': lambda ser: ser.min(),
                 'volume': lambda ser: ser.mean()
                }  
    
    if src_col not in df.columns:
        print('src col not in df columns')
        raise 
    dt_col = '-'.join(['ts',str(scale) ])
    agg_df=pd.DataFrame({})
    
    df[dt_col]=floor_dt(df,'timestamp',scale)
#    df[dt_col]=calculate_fun(df=df,fun_name='floor_dt',str_col='timestamp',scale=scale) # need to write floor column to df  to later groupby it 
    agg_df[dt_col]=df[dt_col].unique().copy()
    for col in cols:
        g=df[[col,dt_col]].groupby([dt_col])
        ser=g.apply(agg_funs_d[col])[col].reset_index(name=col)
        agg_df=agg_df.merge(ser,left_on=dt_col,right_on=dt_col)
    #agg_df['index']=agg_df.index
    return agg_df
    
    



if __name__=='__main__':
    pass
