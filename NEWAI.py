import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# --- Load Data ---
df = pd.read_csv('your_data.csv')

# --- Encode 'Type' ---
le = LabelEncoder()
df['Type_enc'] = le.fit_transform(df['Type'])

# --- Features and targets ---
features = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 
            'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type_enc', 'Cycle']
target_cols = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)']

# --- Normalize features ---
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# --- Create sequences ---
sequence_length = 15

class BatteryDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data.reset_index(drop=True)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        seq_x = self.data.loc[idx:idx+self.seq_length-1, features].values.astype(np.float32)
        seq_y = self.data.loc[idx+self.seq_length, target_cols].values.astype(np.float32)
        return torch.tensor(seq_x), torch.tensor(seq_y)

# --- Split data by cycles ---
train_df = df[df['Cycle'] <= 15]
val_df = df[(df['Cycle'] > 15) & (df['Cycle'] <= 20)]

train_dataset = BatteryDataset(train_df, sequence_length)
val_dataset = BatteryDataset(val_df, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# --- Define LSTM model ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step output
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_dim = len(features)
hidden_dim = 64
output_dim = len(target_cols)

model = LSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
epochs = 30

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_pred = model(X_val)
            loss = criterion(val_pred, y_val)
            val_loss += loss.item() * X_val.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

# --- Predict and plot on validation set ---
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_val, y_val in val_loader:
        pred = model(X_val)
        predictions.append(pred.numpy())
        actuals.append(y_val.numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

plt.figure(figsize=(10,5))
plt.plot(actuals[:,0], label='Real Adjusted Potential')
plt.plot(predictions[:,0], label='Predicted Adjusted Potential')
plt.legend()
plt.show()
