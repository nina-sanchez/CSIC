import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# CHAT AI 


# Load your CSV
df = pd.read_csv("final-data-1302.csv")

# Select relevant columns
features = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 
            'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)']

# Filter for just the first 10 cycles
df = df[df['cycle'] <= 10].reset_index(drop=True)

# Scale all features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Add scaled data back to DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=features)
df_scaled['cycle'] = df['cycle'].values  # keep cycle info

# Define inputs and outputs
input_cols = features
output_cols = ['Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)']

# Group data by cycle and create sequences
sequence_data = []
for cycle_num, group in df_scaled.groupby('cycle'):
    group = group[input_cols].values
    sequence_data.append(group)

# Convert to sequences and targets
X, y = [], []
for seq in sequence_data:
    if len(seq) < 2:
        continue
    X.append(seq[:-1])  # input sequence
    y.append(seq[1:, -3:])  # output = next timestep’s [Freq, Zre, Zim]

# Pad sequences to the same length (use max length)
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

max_len = max([len(x) for x in X])

X_padded = [np.pad(x, ((0, max_len - len(x)), (0, 0)), mode='constant') for x in X]
y_padded = [np.pad(y_, ((0, max_len - len(y_)), (0, 0)), mode='constant') for y_ in y]

X_tensor = torch.tensor(X_padded, dtype=torch.float32)
y_tensor = torch.tensor(y_padded, dtype=torch.float32)

# Create train/test split
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Dataset and Dataloader
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BatteryDataset(X_train, y_train)
val_dataset = BatteryDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = LSTMModel(input_size=len(input_cols), hidden_size=64, output_size=len(output_cols))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    
import matplotlib.pyplot as plt

# Pick one sample from the validation set
model.eval()
with torch.no_grad():
    sample_X = X_val[0].unsqueeze(0)  # add batch dimension
    sample_y_true = y_val[0].numpy()
    sample_y_pred = model(sample_X).squeeze(0).numpy()

# Unscale the data back to original values
# Remember: scaler expects all 5 columns, but we only want to inverse the last 3
# So we’ll pad back to 5 columns with zeros

def inverse_output_scaling(pred, scaler, output_cols, all_cols):
    dummy = np.zeros((pred.shape[0], len(all_cols)))
    for i, col in enumerate(output_cols):
        col_idx = all_cols.index(col)
        dummy[:, col_idx] = pred[:, i]
    return scaler.inverse_transform(dummy)[:, [all_cols.index(col) for col in output_cols]]

# Reverse scaling
pred_unscaled = inverse_output_scaling(sample_y_pred, scaler, output_cols, input_cols)
true_unscaled = inverse_output_scaling(sample_y_true, scaler, output_cols, input_cols)

# Plot predictions vs ground truth for each output
for i, label in enumerate(output_cols):
    plt.figure(figsize=(10, 4))
    plt.plot(true_unscaled[:, i], label=f'True {label}')
    plt.plot(pred_unscaled[:, i], label=f'Predicted {label}')
    plt.title(f'{label} - Predicted vs True')
    plt.xlabel('Timestep')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

