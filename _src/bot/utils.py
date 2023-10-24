import matplotlib.pyplot as plt
import pandas as pd 
import torch 
from matplotlib.ticker import MaxNLocator
import logging 

#from torch_model4 import Network
from torch.utils.data import Dataset, DataLoader, random_split


def plot_two_dfs(top_df, bot_df, top_chart_cols=['col1', 'col2'],top_index=None
                 , bottom_chart_cols=['col1', 'col3'],bot_index=None
                 , additional_sers=None
                 ,do_show=False
                 ,do_save=True
                 ,save_fp='./data/two_dfs.png'):
    N=2
    if len(additional_sers)>2:
        N=len(additional_sers)
    
    fig,ax=plt.subplots(N,1,sharex=False,sharey=False)
    
    df=top_df
    # Top chart
    if top_index is None:
        i=df.index 
    else:
        i=df[top_index]
    
    if top_chart_cols is not None:
        for col in top_chart_cols:
            ax[0].plot(i, df[col], label=col, linestyle='-', marker='x')
        ax[0].legend()
        ax[0].set_title('Top Chart: ' + ', '.join(top_chart_cols))

    df=bot_df
    if bot_index is None:
        i=df.index 
    else:
        i=df[bot_index]
    
    if bottom_chart_cols is not None:
        for col in bottom_chart_cols:
            ax[1].plot(i, df[col], label=col, linestyle='-', marker='.')
        ax[1].legend()
        ax[1].set_title('Bottom Chart: ' + ', '.join(bottom_chart_cols))

    if additional_sers is not None:
        for tup in additional_sers:
            wchich_chart=tup[0]
            index=tup[1]
            value=tup[2]
            msk=value!=0
            index=index[msk]
            value=value[msk]

            linestyle=tup[3]
            marker=tup[4]
            color=tup[5]
            ax[wchich_chart].plot(index,value,linestyle=linestyle, marker=marker, markerfacecolor=color,markeredgecolor=color)

    plt.tight_layout()
    ax[0].grid(True,which="both",ls="-", color='0.65')
    ax[1].grid(True,which="both",ls="-", color='0.65')
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=10))  # Let's say you want 10 ticks for the top plot
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=10))  # And 10 ticks for the bottom plot as well

    if do_show:        
        plt.show()
    if do_save:
        plt.savefig(save_fp)

def plot_df(df, top_chart_cols=['col1', 'col2'], bottom_chart_cols=['col1', 'col3']):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    # Top chart
    if top_chart_cols is not None:
        for col in top_chart_cols:
            ax[0].scatter(df.index, df[col], label=col)
        ax[0].legend()
        ax[0].set_title('Top Chart: ' + ', '.join(top_chart_cols))
    
    
    # Bottom chart
    if bottom_chart_cols is not None:
        for col in bottom_chart_cols:
            ax[1].scatter(df.index, df[col], label=col)
        ax[1].legend()
        ax[1].set_title('Bottom Chart: ' + ', '.join(bottom_chart_cols))
    
    plt.tight_layout()
    ax[0].grid(True,which="both",ls="-", color='0.65')
    ax[1].grid(True,which="both",ls="-", color='0.65')
    plt.show()
    
def plot_candlestick(df
                     , shorts_ser:pd.Series = pd.Series(dtype=float)  # my shorts 
                     , longs_ser:pd.Series = pd.Series(dtype=float)   # my longs 
                     , real_longs:pd.Series = pd.Series(dtype=float)  # model longs 
                     , real_shorts:pd.Series = pd.Series(dtype=float) # model shorts 
                     , purple_ser:pd.Series = pd.Series(dtype=float) # model shorts 
                     , additional_lines = None # top chart additional line 
                     , extra_sers=[]
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
        ax[0].plot(df[msk].index, df[msk]['low']*df[msk]['LONGS_SIGNAL'], '^', color='lightgreen')
    if 'SHORTS_SIGNAL' in df.columns:
        msk=df['SHORTS_SIGNAL']==1
        ax[0].plot(df[msk].index, df[msk]['high']*df[msk]['SHORTS_SIGNAL'],'vr')
        
        
        
    if not shorts_ser.empty:
        mask=shorts_ser>0
        ax[0].plot(shorts_ser.index[mask],shorts_ser[mask],'vr')

    if not longs_ser.empty:
        mask=longs_ser>0
        ax[0].plot(longs_ser.index[mask],longs_ser[mask], '^', color='lightgreen')
        
    if not real_longs.empty:
        ax[0].plot(real_longs.index,real_longs,'og')

    if not real_shorts.empty:
        ax[0].plot(real_shorts.index,real_shorts,'or')
        
    if not purple_ser.empty:
        mask=purple_ser>0
        ax[0].plot(purple_ser.index[mask],purple_ser[mask],'om')
        
    for tup in extra_sers:
        which_chart=tup[0]
        marker=tup[1]
        ser=tup[2]
        mask=ser>0
        ax[which_chart].plot(ser.index[mask],ser[mask],marker)
        
        
    if additional_lines is not None:
        for tup in additional_lines:
            which_chart=tup[0]
            series=tup[1]
            ax[which_chart].plot(series.index,series,'-',label=series.name)
        
    ax[0].legend()
    plt.show()
    return ax 
    
    
    
# evaluate a model on a dataframe containing quantiles
def evaluate_model(model, quantiles_df, signals_df, model_column='model_signal'):
    X = torch.tensor(quantiles_df.values).float()

    val_loader = DataLoader(X, batch_size=32, shuffle=False)  # Make sure shuffle is False
    
    outputs_list = []
    
    for data in val_loader:
        outputs = model(data)
        print(outputs.shape)
        print(outputs)
        exit(1)
        outputs_list.extend(outputs.squeeze().tolist())  # Assuming outputs are 1D tensors for each data point
    signals_df[model_column] = [int(round(i)) for i in    outputs_list ] 
    print(len(signals_df))
    print(len(quantiles_df))
    return signals_df


    # evaluates and plots model with respect to it's training 
def plot_evaluate_model(model, quantiles_df, signals_df,eval_signal ='eval_signal',model_signal='wave_signal'):
    X = torch.tensor(quantiles_df.values).float()
    val_loader = DataLoader(X, batch_size=32, shuffle=False)  # Make sure shuffle is False
    outputs_list = []
    for data in val_loader:
        outputs = model(data)
        outputs_list.extend(outputs.squeeze().tolist())  # Assuming outputs are 1D tensors for each data point
    signals_df[eval_signal] = [int(round(i)) for i in    outputs_list ] 
    
    print(len(signals_df))

    # plot candlestick 
    d=list(signals_df.columns)
    l=[i for i in d if 'q' not in i]
    print(l)

#    signals_df[eval_signal]
    cols=['open','close','low','high','volume',eval_signal,model_signal]
    tmp_df=signals_df[cols]
    model_msk=tmp_df[model_signal]==1
    eval_msk=tmp_df[eval_signal]==1
    extra_sers=( (0, '^', tmp_df[model_msk][model_signal] * tmp_df[model_msk]['open']  , 'o', 'lightgreen', 'lightgreen'),
                (0, 'v', tmp_df[eval_msk][eval_signal] * tmp_df[eval_msk]['close'] , 'o', 'red', 'red') )
    plot_candlestick(df=tmp_df,extra_sers=extra_sers)
    return signals_df







def read_df(df_fp):
    df=pd.read_csv(df_fp,sep='|')
    return df

def write_df(df,name,fp=None,mode='w'):
    if mode=='a':
        header=False
    else:
        header=True
    if fp is None:
        fp='data/'+name+'.csv'
    df.to_csv(fp,sep='|',mode=mode,index=False,header=header)


def append_to_csv(df,fp='./data/test.csv'):
    df.to_csv(fp,sep='|',index=False,mode='a',header=False)



def plot_hist(df,col1,col2,ax1=0,ax2=1):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    df[col1].hist(ax=ax[0], bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax[ax1].set_title('Histogram of Column A')
    ax[ax1].set_ylabel('Count')
    ax[ax1].grid(axis='y')

    # Plot histogram of column 'B' on the second subplot
    df[col2].hist(ax=ax[0], bins=50, edgecolor='black', alpha=0.7, color='red')
    ax[ax2].set_title('Histogram of Column B')
    ax[ax2].set_xlabel('Value')
    ax[ax2].set_ylabel('Count')
    ax[ax2].grid(axis='y')

    plt.tight_layout()
    plt.show()    



def read_model(model_fp,quantiles_df,Network):
    x,y=quantiles_df.shape

    model=Network(y,1,scale_factor=2)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    return model


def setup_logging(name,mode='w',level=20):
    logging.basicConfig(filename=f'./logs/{name}.log',level=level, filemode=mode, format='%(asctime)s %(name)s - %(levelname)s - %(message)s')
    logging.propagate = False

def log_stuff(msg='',level=20,**kwargs ) :
    ss='\n'
    if len(kwargs.keys())>1:
        ss='\n'
    s=f'{ss}    '+ '\n    '.join([f'{k} : {v}' for k,v in kwargs.items()])
    msg=msg+s
    logging.log(level,msg)


def setup_logging2(name, mode='w', level=20):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(f'./logs/{name}.log', mode)
    formatter = logging.Formatter('%(asctime)s %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

def log_stuff2(logger,msg='',level=20, **kwargs ) :
    ss='\n'
    if len(kwargs.keys())>1:
        ss='\n'
    s=f'{ss}    '+ '\n    '.join([f'{k} : {v}' for k,v in kwargs.items()])
    msg=msg+s
    logger.log(level,msg)


def ar_torch_evaluate(ar : list, Network ):
    # 
    model=Network(len(ar),1,scale_factor=2)
    pass 


if __name__=='__main__':
    df=read_df('./data/test.csv')
    #append_to_csv(df=df)
    write_df(df=df,name='test',mode='a')