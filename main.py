# putting things together so i dont forget them 
from utils import Utils as u 
from backtest_score_algos import lookahead_score,eval_strategy


# adding labels ( points with roi > 1 ) to a df 
def backtest_score_workflow():
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
 
 
backtest_score_workflow()