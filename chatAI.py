import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ======================
# 1. Data Loading + Cleaning
# ======================

# User selects CSV file
Tk().withdraw()
file_path = askopenfilename(title="select a .csv file", filetypes=[("CSV files", "*.csv")])
if not file_path:
    print("no file selected.")
    exit()
try:
    data = pd.read_csv(file_path, delimiter=",")
    print(f"file loaded: {file_path}")
except Exception as e:
    print(f"error loading file: {e}")
    exit()

data = data[['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']]
data['Potential (V)'] = data['Adjusted Potential (V)']
data['Capacity (mAh/g)'] = data['Adjusted Relative Capacity (mAh/g)']
data = data[['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']]

# ======================
# 2. Define Input and Output Features
# ======================

# Input features (all the columns except the output ones)
input_features = ['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']

# Output features (we only want to predict these columns)
output_features = ['Potential (V)', 'Capacity (mAh/g)']  # Adjusted here

# Seperating data into input and output
data_input = data[input_features]
data_output = data[output_features]

# Scaling data
scaler_input = MinMaxScaler()
data_input_scaled = scaler_input.fit_transform(data_input)

scaler_output = MinMaxScaler()
data_output_scaled = scaler_output.fit_transform(data_output)

# Combine input and output data for sequence creation
scaled_data = pd.DataFrame(np.hstack([data_input_scaled, data_output_scaled]), columns=input_features + output_features)

# ======================
# 3. Sequence Creation
# ======================

def create_sequences(data, sequence_length=10):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-2])  # Adjusted: Use all input features (last 2 columns are output)
        y.append(data[i + sequence_length, -2:])  # Predict only the output columns (last 2 columns)
    return np.array(X), np.array(y)

# Create sequences for training
sequence_length = 10  # Sequence length is 10 cycles
X, y = create_sequences(scaled_data.values, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# ======================
# 4. Build Model
# ======================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out

# ======================
# 5. Train Model
# ======================

# Hyperparameters
input_size = 7  # Number of input features
hidden_size = 64  # LSTM hidden units
output_size = 2  # Number of output features (Adjusted Potential and Capacity)
num_epochs = 50  # Number of training epochs
batch_size = 32  # Batch size
learning_rate = 0.001  # Learning rate

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients
    outputs = model(X_tensor)  # Forward pass
    loss = criterion(outputs, y_tensor)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update model weights
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ======================
# 6. Evaluate Model
# ======================

model.eval()  # Switch to evaluation mode

with torch.no_grad():
    predictions = model(X_tensor)

# Rescale the predictions back to the original scale
predictions_rescaled = scaler_output.inverse_transform(predictions.numpy())

# Plot actual vs predicted values for "Adjusted Potential (V)" and "Adjusted Relative Capacity (mAh/g)"
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(y_tensor[:, 0].numpy(), label="Actual Potential (V)")
plt.plot(predictions_rescaled[:, 0], label="Predicted Potential (V)")
plt.title("Actual vs Predicted Potential (V)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_tensor[:, 1].numpy(), label="Actual Capacity (mAh/g)")
plt.plot(predictions_rescaled[:, 1], label="Predicted Capacity (mAh/g)")
plt.title("Actual vs Predicted Capacity (mAh/g)")
plt.legend()

plt.show()
