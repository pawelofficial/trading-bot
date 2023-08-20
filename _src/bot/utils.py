import matplotlib.pyplot as plt
import pandas as pd 
import torch 
from matplotlib.ticker import MaxNLocator

from torch_model4 import Network
from torch.utils.data import Dataset, DataLoader, random_split


def plot_two_dfs(top_df, bot_df, top_chart_cols=['col1', 'col2'],top_index=None
                 , bottom_chart_cols=['col1', 'col3'],bot_index=None
                 , additional_sers=None):
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

    
    plt.show()

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

    
def old_plot_candlestick(df
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

    fig,ax=plt.subplots(2,1,sharex=True,sharey=True)
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
        label=tup[3]
        mask=ser>0
        ax[which_chart].plot(ser.index[mask],ser[mask],marker,label=label)
        
        
    if additional_lines is not None:
        for tup in additional_lines:
            which_chart=tup[0]
            series=tup[1]
            ax[which_chart].plot(series.index,series,'-',label=series.name)
        
    ax[0].legend()
    plt.show()
    return ax 
    
    
    
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
        outputs_list.extend(outputs.squeeze().tolist())  # Assuming outputs are 1D tensors for each data point
    signals_df[model_column] = [int(round(i)) for i in    outputs_list ] 
    return signals_df

def read_df(df_fp):
    df=pd.read_csv(df_fp,sep='|')
    return df



def read_model(model_fp,quantiles_df):
    x,y=quantiles_df.shape

    model=Network(y,1,scale_factor=2)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    return model