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

if __name__=='__main__':
    import logging
    logging.basicConfig(filename='./src/logs/wave_model.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.warning('starting training')

cool_name_for_variable='BTC-USD2022-01-01_2023-01-01'
raw_fps=f'./src/data_backups/{cool_name_for_variable}.csv'

df=pd.read_csv(raw_fps)

df=wf__aggregate_df(df=df,scale=15)
LR=0.0001
epochs=1000
nlq_number=50
nlq_steepness=15
nlq_accuracy=5
qdf,bdf=wf__make_quantiles_df(input_df=df,nlq_number=nlq_number,nlq_steepness=nlq_steepness,nlq_accuracy=nlq_accuracy)
print(qdf.shape)

y,_=wf__make_signals(input_df=df)
do_plot=False
if do_plot: # lets take a look at signal pytorch will be fitting into 
    from indicators import plot_candlestick
    df=df.iloc[:1000]
    msk=y==1
    longs_ser=df['low'][msk]
    ema10=_['ema10']
    ema15=_['ema15']
    plot_candlestick(df=df,longs_ser=longs_ser,additional_lines=((0,ema10),(0,ema15)   )  )
    exit(1)


Y=y
X=qdf

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, X,Y):
        self.x = torch.tensor(X.values).float()
        self.y = torch.tensor(Y.values).float()
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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
if 0: # load model if you have it and has the same params 
    m='coinbase_model_202307011741'
    path=f'./src/models/{m}.pth'
    model.load_state_dict(torch.load(path))
from torch import optim

# Define loss function and optimizerkcaE&(U-)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)                                # lr 

# Define number of epochs


# Split into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
ts_now=time.time()
try:
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

        if epoch % 100 ==0:
            print('saving model !')
            path=f'./src/models/wave_models/wave_loop.pth'
            torch.save(model.state_dict(), path)
            elapsed_time = time.time() - ts_now
            print(f'elapsed time {elapsed_time}')

        print(f"Epoch {epoch+1} - Training loss: {running_loss/len(train_loader)}")
except KeyboardInterrupt:
    pass 

ts=datetime.datetime.now().strftime("%Y%m%d%H%M")
path=f'./src/models/wave_models/wave_{ts}.pth'
torch.save(model.state_dict(), path)

model = Network(in_features,1,scaling_factor)

# Load
model.load_state_dict(torch.load(path))
model.eval() # Set model to evaluation mode
# Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
val_loss = 0
true_positives = 0
false_negatives=0
total = 0
correct=0
TPS,TNS,FPS,FNS=0,0,0,0
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
        TPS += (predicted == labels == 1).sum().item()    # True Positives
        TNS += (predicted == labels == 0).sum().item()    # True Negatives
        FPS += (predicted == 1 & labels == 0).sum().item()  # False Positives
        FNS += (predicted == 0 & labels == 1).sum().item()  # False Negatives
        
        true_positives += (predicted == labels==1).sum().item()
        false_negatives += (predicted == labels==0).sum().item()
        correct+= (predicted == labels).sum().item()
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
file : {raw_fps}
epochs : {epochs}
nlq_number {nlq_number}
nlq_steepness {nlq_steepness}
nlq_accuracy {nlq_accuracy}
signal model : wave_model
TRUE POSITIVES ( 1,1 ) TP : {TPS} 
TRUE NEGATIVES ( 0,0 ) TN : {TNS}
FALSE NEGATIVES (0,1) FN : {FNS}
FALSE POSITIVES (1,0) FP : {FPS}       - bad one 
"""
print(s)


logging.warning(f'------------{ts}--------------')
logging.warning(s)