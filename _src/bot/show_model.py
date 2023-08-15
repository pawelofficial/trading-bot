import pandas as pd 
import torch 
from torch_model4 import Network
from torch.utils.data import Dataset, DataLoader, random_split
from utils import plot_candlestick
from rich.traceback import install
install()


def read_df(df_fp):
    df=pd.read_csv(df_fp,sep='|')
    return df

def read_model(model_fp,quantiles_df):
    x,y=quantiles_df.shape

    model=Network(y,1,scale_factor=2)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    return model

# evaluate a model on a dataframe containing quantiles
def evaluate_model(model, quantiles_df, signals_df, model_column='model_output'):
    X = torch.tensor(quantiles_df.values).float()
    val_loader = DataLoader(X, batch_size=32, shuffle=False)  # Make sure shuffle is False
    
    outputs_list = []
    
    for data in val_loader:
        outputs = model(data)
        outputs_list.extend(outputs.squeeze().tolist())  # Assuming outputs are 1D tensors for each data point
    signals_df[model_column] = [int(round(i)) for i in    outputs_list ] 
    return signals_df

# evaluate again a dataframe containing model output and signal output 
def evaluate_df(df,model_col='model_output',signal_col='wave_signal'):
    TP=0 # 1 1 
    TN=0 # 0 0
    FP=0 # 1 0
    FN=0 # 0 1
    
    for no,row in df.iterrows():
        d=row.to_dict()
        model_output=d[model_col] #    int(round(d['model_output']))
        signal=d[signal_col]        #   int(round(d['wave_signal']))
        if model_output==1 and signal==1:
            TP+=1
        elif model_output==0 and signal==0:
            TN+=1
        elif model_output==1 and signal==0:
            FP+=1
        elif model_output==0 and signal==1:
            FN+=1
    print(f'TP: {TP} --> 1 1 ')
    print(f'TN: {TN} --> 0 0')
    print(f'FP: {FP} --> 1 0 bad one ')
    print(f'FN: {FN} --> 0 1 ')
    print(f'len df -> {len(df)} ')
    return None


# reads signal df 
# reads pytorch model 
# evaluates pytorch model 
# plots stuff so you can see it 

if __name__=='__main__':
    signals_fp='./data/signals_df.csv'
    quantiles_fp='./data/quantiles_df.csv'
    model_fp='./models/wave_models/wave_loop.pth'
    
    signals_df=read_df(signals_fp)

    
    quantiles_df=read_df(quantiles_fp)
    model=read_model(model_fp,quantiles_df)


    df=evaluate_model(model,quantiles_df,signals_df )
    n=len(df)//2
    df=df[-n:]
    
    wave_signal=df['wave_signal']*df['open']
    model_output=df['model_output']*df['open']
    extra_sers=((0,'*c',wave_signal,'wave') 
                ,(1,'.r',model_output,'pytorch'),(1,'--',df['open'],'open')   )
    
    plot_candlestick(df
                     ,extra_sers=extra_sers
                     )
    