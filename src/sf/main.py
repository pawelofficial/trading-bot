import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import make_classification
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming your model, optimizer, and loss function are defined as below:
# loss_fn = ...


def load_data():
    X, Y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    X = pd.DataFrame(X)
    Y = pd.Series(Y)

    return X,Y

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X.values).float()
        self.y = torch.tensor(Y.values).float().unsqueeze(1)  # Add an extra dimension


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

# Let's assume we train for 10 epochs
BATCH_SIZE = 32
n_epochs = 10

def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader


X, Y = load_data()
print(X.shape)

dataset = CustomDataset(X, Y)
train_dataset, val_dataset = split_dataset(dataset)
train_loader,test_loader=create_dataloaders(train_dataset, val_dataset)

model = Network(X.shape[1], 1, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


for epoch in range(n_epochs):
    
    # Training Phase 
    model.train()
    
    for batch in train_loader:
        # Assuming your batch is a tuple (inputs, targets)
        inputs, targets = batch
        
        # Move the training data to the GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Evaluation Phase
    model.eval()
    
    with torch.no_grad():   # Deactivate autograd engine to reduce memory usage and speed up computations
        total = 0
        correct = 0
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            predicted = (outputs.data > 0.5).float()
            
            # Calculate accuracy
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        print('Accuracy of the model on the test data: {}%'.format(100 * correct / total))

    torch.save(model.state_dict(), "model.pth")