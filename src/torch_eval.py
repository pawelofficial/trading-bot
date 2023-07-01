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
        def __init__(self):
            super(Network, self).__init__()
            self.layer1 = nn.Linear(800, 400)  # Assuming df has 801 features
            self.layer2 = nn.Linear(400, 200)  # Assuming df has 801 features
            self.layer3 = nn.Linear(200, 100)  # Assuming df has 801 features
            self.layer4 = nn.Linear(100, 50)  # Assuming df has 801 features
            self.layer5 = nn.Linear(50, 50)  # Assuming df has 801 features
            self.layer6 = nn.Linear(50, 50)  # Assuming df has 801 features
            self.layer7 = nn.Linear(50, 25)  # Assuming df has 801 features
            self.layer8 = nn.Linear(25, 5)
            self.output = nn.Linear(5, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = nn.functional.relu(self.layer1(x))
            x = nn.functional.relu(self.layer2(x))
            x = nn.functional.relu(self.layer3(x))
            x = nn.functional.relu(self.layer4(x))
            x = nn.functional.relu(self.layer5(x))
            x = nn.functional.relu(self.layer6(x))
            x = nn.functional.relu(self.layer7(x))
            x = nn.functional.relu(self.layer8(x))
            x = self.sigmoid(self.output(x))
            return x

data_fps='./src/data_backups/BTC-USD2022-01-01_2022-02-03.csv'
data_df=pd.read_csv(data_fps)
mth=myMath()
data_df=mth.aggregate(df=data_df, scale =15, src_col='timestamp')
indicators_df=pd.read_csv('./src/data/indicators_BTC-USD2022-01-01_2022-02-03.csv')
green_labels=indicators_df['green']
indicators_df.drop(columns=['green'], inplace=True)


model=Network()
mdl='coinbase_model_202306291136'
path=f'./src/models/models_backups/{mdl}.pth'
dump_fps=f'./src/data/dump.csv'

model.load_state_dict(torch.load(path))
model.eval() # Set model to evaluation mode


input_ts=torch.tensor(indicators_df.values).float()
out=model(input_ts)

out_list = [int(np.round(item,1)) for sublist in out.tolist() for item in sublist]

data_df['green_labels']=green_labels
data_df['green_preds']=out_list
print(data_df['green_labels'].value_counts())
print(data_df['green_preds'].value_counts())  # EXPECTED 0S AND 1S ONLY got wierd values 


true_positives=0
true_negatives=0
false_positives=0
false_negatives=0
for no,row in data_df.iterrows():
    d=row.to_dict()
    
    if d['green_labels']==d['green_preds']==1:
        true_positives+=1
    if d['green_labels']==d['green_preds']==0:
        true_negatives+=1
    if d['green_labels']==1 and d['green_preds']==0:
        true_negatives+=1
    if d['green_labels']==0 and d['green_preds']==1:
        false_negatives+=1
        
print(f'True positives: {true_positives}')
print(f'True negatives: {true_negatives}')
print(f'False positives: {false_positives}')
print(f'False negatives: {false_negatives}')


data_df.to_csv(dump_fps)