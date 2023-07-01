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

logging.basicConfig(filename='./src/logs/model2.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('starting training')


#df,logit_cols,y_col= wf__make_indicators(df_fps='./src/data/data.csv'
#                                         ,out_fps='./src/data/indicators_df.csv')
#
#X = df[logit_cols]  # 801 x columns 
#Y = y_col           # 1 output column 


qdf_fps='./src/data/quantiles_df_15_60_3_months.csv'
edf_fps='./src/data/edf_df_15_60_3_months.csv'
X=pd.read_csv(qdf_fps, index_col=0)
Y=pd.read_csv(edf_fps,index_col=0)['green']


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, X,Y):
        self.x = torch.tensor(X.values).float()
        self.y = torch.tensor(Y.values).float()
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class simpleNetwork(nn.Module):
    def __init__(self,n_features):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_features, 1)  # Assuming df has 801 features
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = self.sigmoid(x)  # directly use sigmoid on the result of the previous layer
        return x

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


# Create a dataset from your dataframe
dataset = CustomDataset(X,Y)

in_features=X.shape[1]
scaling_factor=4
model = Network(in_features,1,scaling_factor)

from torch import optim

# Define loss function and optimizerkcaE&(U-)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)                                # lr 

# Define number of epochs
epochs = 100 # 10-20 is a good start                                               # epochs 

# Split into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Training loss: {running_loss/len(train_loader)}")
    
ts=datetime.datetime.now().strftime("%Y%m%d%H%M")
path=f'./src/models/coinbase_model_{ts}.pth'
torch.save(model.state_dict(), path)

model = Network(in_features,1,scaling_factor)

# Load
model.load_state_dict(torch.load(path))
model.eval() # Set model to evaluation mode
# Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
val_loss = 0
correct = 0
total = 0
predictions = []
true_labels = []
with torch.no_grad(): # We don't need gradient computation in evaluation phase
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # Threshold outputs at 0.5
        predicted = (outputs >= 0.5).float()
        # Save predictions and true labels for later use
        predictions.extend(predicted.view(-1).tolist())
        true_labels.extend(labels.view(-1).tolist())
        
        # Compare with labels to check correctness
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Compute average validation loss
avg_val_loss = val_loss / len(val_loader)

# Compute validation accuracy
val_accuracy = correct / total

# Compute F1 score
f1 = f1_score(true_labels, predictions)

s=f"""
f"Average validation loss: {avg_val_loss}
f"Validation accuracy: {val_accuracy}
f"F1 score: {f1}
Number of 1s in true labels: {true_labels.count(1)}
Number of 0s in true labels: {true_labels.count(0)}
Number of 1s in predictions: {predictions.count(1)}
Number of 0s in predictions: {predictions.count(0)}
true positives : {predictions.count(1)} / {true_labels.count(1)} = {round(predictions.count(1)/true_labels.count(1),2)}
false negatives: {predictions.count(0)} / {true_labels.count(0)} = {round(predictions.count(0)/true_labels.count(0),2)}
Epochs: {epochs}
model: {path}
value counts : {Y.value_counts()}
X shape: {X.shape}
file : {qdf_fps}
"""
print(s)


logging.warning(f'------------{ts}--------------')
logging.warning(s)