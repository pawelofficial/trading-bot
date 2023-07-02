import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd 
from workflows import * 
import datetime
import logging 

class Network(nn.Module):
    def __init__(self, input_features, output_features, scale_factor):
        super(Network, self).__init__()

        layers = []
        in_features = input_features
        while in_features > output_features*scale_factor:
            out_features = max(int(in_features/scale_factor), output_features*scale_factor)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features

        layers.append(nn.Linear(in_features, output_features))
        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = self.sigmoid(x)
        return x

edf=pd.read_csv('./src/data/edf_3months_s10_a3_N50.csv')
qdf=pd.read_csv('./src/data/qdf_3months_s10_a3_N50.csv',index_col=0)


in_features = qdf.shape[1]
scaling_factor = 4
model = Network(in_features,1,scaling_factor)
mdl='coinbase_model_loop'
path=f'./src/models/models_backups/{mdl}.pth'
dump_fps=f'./src/data/dump.csv'

model.load_state_dict(torch.load(path))
model.eval() # Set model to evaluation mode


input_ts=torch.tensor(qdf.values).float()
out=model(input_ts)

out_list = [int(np.round(item,1)) for sublist in out.tolist() for item in sublist]

dump_df=pd.DataFrame()
dump_df['actual_labels']=edf['green']
dump_df['timestamp']=edf['ts-15']
dump_df['pred_labels']=out_list


true_positives=0
true_negatives=0
false_positives=0
false_negatives=0
for no,row in dump_df.iterrows():
    d=row.to_dict()
    
    if d['actual_labels']==d['pred_labels']==1:
        true_positives+=1
    if d['actual_labels']==d['pred_labels']==0:
        true_negatives+=1
    if d['actual_labels']==1 and d['pred_labels']==0:
        true_negatives+=1
    if d['actual_labels']==0 and d['pred_labels']==1:
        false_negatives+=1
        
print(f'True positives: {true_positives}')
print(f'True negatives: {true_negatives}')
print(f'False positives: {false_positives}')
print(f'False negatives: {false_negatives}')


#data_df.to_csv(dump_fps)