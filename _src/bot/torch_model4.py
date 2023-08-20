import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import f1_score
import pandas as pd 
import datetime
import logging 
import time

import wokrflows as wf 




# Set up logging
def setup_logging():
    logging.basicConfig(filename='./logs/wave_model.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
    logging.warning('starting training')

# Load and preprocess the data
#def load_data():
#    df = pd.read_csv(FILE_PATH)
#    df = wf__aggregate_df(df=df, scale=15)
#    qdf, bdf = wf__make_quantiles_df(input_df=df, nlq_number=NLQ_NUMBER, nlq_steepness=NLQ_STEEPNESS, nlq_accuracy=NLQ_ACCURACY)
#    y, _ = wf__make_signals(input_df=df)
#    return qdf, y

#def load_data():
#    fp=wf.wf__download_data()
#    q_df,q_fp,i_df,i_fp,s_df,signal=wf.wf__prep_data(fp=fp)
#    return q_df, signal

# loads signals_Df and quantiles_df 
def load_data(n=None):
    q_df=pd.read_csv('./data/quantiles_df.csv',sep='|')
    s_df=pd.read_csv('./data/signals_df.csv',sep='|')
    if n is None:
        return q_df,s_df['wave_signal']
    else:
        return q_df.iloc[:n,:],s_df['wave_signal'][:n]

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
def split_dataset(dataset,N=1):
    train_size = int(1 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

# Create DataLoaders
def create_dataloaders(train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader

# Train the model
def train_model(train_loader, model, criterion, optimizer, epochs,loop_model=f'./models/wave_models/wave_loop.pth'):
    ts_now = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss= my_bceloss(outputs, labels.unsqueeze(1))
            #loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 100 == 0:
            print(f'saving model !  {loop_model} ')
            torch.save(model.state_dict(), loop_model)
            elapsed_time = time.time() - ts_now
            print(f'elapsed time {elapsed_time}')

        print(f"Epoch {epoch+1} - Training loss: {running_loss/len(train_loader)}")

# Save the trained model
def save(model,save_fps):

    torch.save(model.state_dict(), save_fps)
    return save_fps

    # bceloss with penalty for false positives ! 
def my_bceloss(outputs, labels, base_weights=None, false_positive_penalty=10.0):
    # BCE loss implementation 
    epsilon = 1e-12
    loss = -labels * torch.log(outputs + epsilon) - (1 - labels) * torch.log(1 - outputs + epsilon)

    # Calculate false positive penalty
    fp_penalty = outputs * (1 - labels)
    fp_weights = 1 + fp_penalty * (false_positive_penalty - 1)

    # Calculate false positive penalty
    #fp_penalty = ((outputs >= 0.5).float() * (1 - labels)).detach()
    #fp_weights = 1 + fp_penalty * (false_positive_penalty - 1)

    # Combine with base weights
    if base_weights is not None:
        combined_weights = base_weights * fp_weights
    else:
        combined_weights = fp_weights

    loss = combined_weights * loss

    return loss.mean()

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
            #loss = criterion(outputs, labels)
            loss=my_bceloss(outputs,labels.unsqueeze(1))
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

def log_results(avg_val_loss, val_accuracy, f1, predictions, true_labels, TPS, TNS, FPS, FNS, trues, falses, model_path, X, Y
                , epochs, nlq_number, nlq_steepness, nlq_accuracy, data_fps,comment=''):
    
    # Header
    header = f"------------{datetime.datetime.now().strftime('%Y%m%d%H%M')}--------------"
    
    # Model Info
    model_info = f"""
    Model Info:
    - Model: wave_model
    - Model Path: {model_path}
    - Epochs: {epochs}
    """
    
    # Data Info
    data_info = f"""
    Data Info:
    - X Shape: {X.shape}
    - Value Counts: {Y.value_counts()}
    - File Path: {data_fps}
    """
    
    # Training Info
    training_info = f"""
    Training Info:
    - NLQ Number: {nlq_number}
    - NLQ Steepness: {nlq_steepness}
    - NLQ Accuracy: {nlq_accuracy}
    """
    
    # Metrics
    metrics = f"""
    Metrics:
    - Average Validation Loss: {avg_val_loss:.3f}
    - Validation Accuracy: {val_accuracy:.3f}
    - F1 Score: {f1:.3f}
    - TRUE POSITIVES (1,1) TP:  {TPS} ({round(TPS/trues,5)})
    - TRUE NEGATIVES (0,0) TN:  {TNS} ({round(TNS/falses,5)})
    - FALSE NEGATIVES (0,1) FN: {FNS}
    - FALSE POSITIVES (1,0) FP: {FPS} (bad one)
    - Number_of_1s_in_true_labels: {true_labels.count(1)}
    - Number_of_0s_in_true_labels: {true_labels.count(0)}
    - Number_of_1s_in_predictions: {predictions.count(1)}
    - Number_of_0s_in_predictions: {predictions.count(0)}
    - Dataset Size : {len(predictions)}
    - comment : {comment}
    """
    
    # Construct the log message
    log_msg = f"{header}\n{model_info}\n{data_info}\n{training_info}\n{metrics}"
    
    # Print and log the message
    print(log_msg)
    logging.warning(log_msg)



if __name__ == '__main__':
# Constants
    loop_model=f'./models/wave_models/wave_loop.pth'                        # model to be saved during loop 
    preload_model='./models/wave_models/wave_loop.pth'               # model to be preloaded before training 
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")                     # model to be saved 
    save_model = f'./models/wave_models/wave_{ts}.pth'
    LR = 0.0001
    EPOCHS = 101
    TRAIN_TEST_SPLIT=1 
    NLQ_NUMBER = 50
    NLQ_STEEPNESS = 15
    NLQ_ACCURACY = 5
    FILE_NAME = 'BTC-USD2022-01-01_2022-02-03'
    FILE_PATH = f'./data/data_backups/{FILE_NAME}.csv'
    BATCH_SIZE = 32
    SCALING_FACTOR = 2
    setup_logging()
    X, Y = load_data()
    dataset = CustomDataset(X, Y)
    model = Network(X.shape[1], 1, SCALING_FACTOR)
    model.load_state_dict(torch.load(preload_model))
    # add weights to criterion
    comment = ' bce loss explicit formula with weights added to false positives'
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_dataset, val_dataset = split_dataset(dataset,N= TRAIN_TEST_SPLIT)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
    train_model(train_loader, model, criterion, optimizer, EPOCHS,loop_model=loop_model)
    
    if TRAIN_TEST_SPLIT ==1: 
        val_loader=train_loader
        
    avg_val_loss, val_accuracy, f1, predictions, true_labels, TPS, TNS, FPS, FNS, trues, falses = evaluate_model(val_loader, model, criterion)
    model_path = save(model,save_model)
    log_results(avg_val_loss, val_accuracy, f1, predictions, true_labels, TPS, TNS, FPS, FNS, trues, falses,model_path, X, Y,
                EPOCHS,NLQ_NUMBER,NLQ_STEEPNESS,NLQ_ACCURACY,FILE_PATH,comment=comment)
    
   
