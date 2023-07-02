import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import f1_score
import pandas as pd 
from workflows import * 
import datetime
import logging 
import time

# Constants
LR = 0.0001
EPOCHS = 10000
NLQ_NUMBER = 50
NLQ_STEEPNESS = 15
NLQ_ACCURACY = 5
FILE_NAME = 'BTC-USD2022-01-01_2022-02-03'
FILE_PATH = f'./src/data_backups/{FILE_NAME}.csv'
BATCH_SIZE = 32
SCALING_FACTOR = 2

# Set up logging
def setup_logging():
    logging.basicConfig(filename='./src/logs/wave_model.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
    logging.warning('starting training')

# Load and preprocess the data
def load_data():
    df = pd.read_csv(FILE_PATH)
    df = wf__aggregate_df(df=df, scale=15)
    qdf, bdf = wf__make_quantiles_df(input_df=df, nlq_number=NLQ_NUMBER, nlq_steepness=NLQ_STEEPNESS, nlq_accuracy=NLQ_ACCURACY)
    y, _ = wf__make_signals(input_df=df)
    return qdf, y

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X.values).float()
        self.y = torch.tensor(Y.values).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Network Class
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

# Split dataset into train and validation
def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

# Create DataLoaders
def create_dataloaders(train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader

# Train the model
def train_model(train_loader, model, criterion, optimizer, epochs):
    ts_now = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 100 == 0:
            print('saving model !')
            path = f'./src/models/wave_models/wave_loop.pth'
            torch.save(model.state_dict(), path)
            elapsed_time = time.time() - ts_now
            print(f'elapsed time {elapsed_time}')

        print(f"Epoch {epoch+1} - Training loss: {running_loss/len(train_loader)}")

# Save the trained model
def save_model(model):
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    path = f'./src/models/wave_models/wave_{ts}.pth'
    torch.save(model.state_dict(), path)
    return path

# Evaluate the model
def evaluate_model(val_loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_loss = 0
    TPS, TNS, FPS, FNS = 0, 0, 0, 0
    trues, falses = 0, 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            predictions.extend(predicted.view(-1).tolist())
            true_labels.extend(labels.view(-1).tolist())
            TPS += ((predicted == 1) & (labels == 1)).sum().item()    # True Positives
            TNS += ((predicted == 0) & (labels == 0)).sum().item()    # True Negatives
            FPS += ((predicted == 1) & (labels == 0)).sum().item()    # False Positives
            FNS += ((predicted == 0) & (labels == 1)).sum().item()    # False Negatives
            trues +=(labels==1).sum().item()
            falses +=(labels==0).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = (TPS + TNS) / (TPS + TNS + FPS + FNS)
    f1 = f1_score(true_labels, predictions)
    return avg_val_loss, val_accuracy, f1, predictions, true_labels, TPS, TNS, FPS, FNS, trues, falses

# Print and log results
def log_results(avg_val_loss, val_accuracy, f1, predictions, true_labels, TPS, TNS, FPS, FNS, trues, falses, model_path, X, Y):
    s = f"""[...]
    """
    print(s)
    logging.warning(f'------------{datetime.datetime.now().strftime("%Y%m%d%H%M")}--------------')
    logging.warning(s)

if __name__ == '__main__':
    setup_logging()
    X, Y = load_data()
    dataset = CustomDataset(X, Y)
    model = Network(X.shape[1], 1, SCALING_FACTOR)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
    train_model(train_loader, model, criterion, optimizer, EPOCHS)
    model_path = save_model(model)
   
