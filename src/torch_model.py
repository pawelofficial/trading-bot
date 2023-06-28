import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd 
from workflows import * 


#df,logit_cols,y_col= wf__make_indicators(df_fps='./src/data/data.csv'
#                                         ,out_fps='./src/data/indicators_df.csv')
#
#X = df[logit_cols]  # 801 x columns 
#Y = y_col           # 1 output column 


df=pd.read_csv('./src/data/indicators_df.csv')
Y=df['green']
print(Y.value_counts())
df=df.drop(columns=['green'])
X=df



# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, X,Y):
        self.x = torch.tensor(X.values).float()
        self.y = torch.tensor(Y.values).float()
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if 0: # 0 -> use simple model, 1 use complex model
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.layer1 = nn.Linear(800, 1)  # Assuming df has 801 features
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = nn.functional.relu(self.layer1(x))
            x = self.sigmoid(x)  # directly use sigmoid on the result of the previous layer
            return x
else:
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


# Create a dataset from your dataframe
dataset = CustomDataset(X,Y)
model = Network()

from torch import optim

# Define loss function and optimizer
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
    
path='./src/models/coinbase_model.pth'
torch.save(model.state_dict(), path)

model = Network()

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

print(f"Average validation loss: {avg_val_loss}")
print(f"Validation accuracy: {val_accuracy}")
print(f"F1 score: {f1}")

# Print the distribution of true labels and predictions
print(f"Number of 1s in true labels: {true_labels.count(1)}")
print(f"Number of 0s in true labels: {true_labels.count(0)}")
print(f"Number of 1s in predictions: {predictions.count(1)}")
print(f"Number of 0s in predictions: {predictions.count(0)}")