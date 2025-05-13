# LSTM AI
# I am using LSTM (long short-term memory) model, which is a type
# of recurrent neural network (RNN) designed to handle sequential
# data - such as time series or cycles of data. LSTM remembers
# information over time, so it can predict the next step in a sequence
# based on patterns from previous cycles. so LSTM here will learn 
# how cycles evolve over time from one cycle to cycle n. it will
# use that cycle hustory to predict future cycles

# more information on this:
# https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/
# referenced this code: https://www.geeksforgeeks.org/long-short-term-memory-networks-using-pytorch/


# reference this for similiar setup --> tabular data
# https://www.kaggle.com/code/alishaangdembe/time-series-forecasting-lstm-hyperparameter-tune

# stackoverflow of someones code --> look for values needed
# https://stackoverflow.com/questions/77007252/how-to-perform-hyperparameter-tuning-of-lstm-using-gridsearchcv



# CHAT CODE
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

# Clean and adjust column names
data = data[['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']]
data['Potential (V)'] = data['Adjusted Potential (V)']
data['Capacity (mAh/g)'] = data['Adjusted Relative Capacity (mAh/g)']
data = data[['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']]

# Input and output features
input_features = ['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
output_features = ['Potential (V)', 'Capacity (mAh/g)']  # Change to these columns for prediction

# Separate the input and output data
data_input = data[input_features]
data_output = data[output_features]

# Scale the input and output data
scaler_input = MinMaxScaler()
data_input_scaled = scaler_input.fit_transform(data_input)

scaler_output = MinMaxScaler()
data_output_scaled = scaler_output.fit_transform(data_output)

# Combine input and output data
scaled_data = pd.DataFrame(np.hstack([data_input_scaled, data_output_scaled]), columns=input_features + output_features)


# ======================
# 2. Sequence Creation
# ======================
def create_sequences(data, sequence_length=10):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-2])  # Use all input features (last 2 columns are output)
        y.append(data[i + sequence_length, -2:])  # Predict the last 2 columns (Adjusted Potential and Capacity)
    return np.array(X), np.array(y)

# Create sequences for training
sequence_length = 10  # You can change this based on your data and how far back you want to look
X, y = create_sequences(scaled_data.values, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# ======================
# 3. Build Model
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]  # Only care about the output of the last time step
        out = self.fc(last_time_step)
        return out


# ======================
# 4. Train Model
# ======================
# Hyperparameters
input_size = 7  # Number of input features at each time step
hidden_size = 64  # Number of hidden units in LSTM
output_size = 2  # Adjusted to 2 for Potential and Capacity
num_epochs = 50  # Number of training epochs
batch_size = 32  # Batch size for training
learning_rate = 0.001  # Learning rate

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
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
# 5. Evaluate Model
# ======================
# Switch to evaluation mode
model.eval()

# Make predictions on the training data
with torch.no_grad():
    predictions = model(X_tensor)

# Rescale the predictions to original scale
predictions_rescaled = scaler_output.inverse_transform(predictions.numpy())

# Plot actual vs predicted values for Adjusted Potential (V) and Adjusted Relative Capacity (mAh/g)
plt.figure(figsize=(12, 6))

# Plot Adjusted Potential (V)
plt.subplot(1, 2, 1)
plt.plot(y_tensor[:, 0].numpy(), label="Actual Potential (V)")
plt.plot(predictions_rescaled[:, 0], label="Predicted Potential (V)")
plt.title("Actual vs Predicted Potential (V)")
plt.legend()

# Plot Adjusted Capacity (mAh/g)
plt.subplot(1, 2, 2)
plt.plot(y_tensor[:, 1].numpy(), label="Actual Capacity (mAh/g)")
plt.plot(predictions_rescaled[:, 1], label="Predicted Capacity (mAh/g)")
plt.title("Actual vs Predicted Capacity (mAh/g)")
plt.legend()

plt.show()











# MY CODE
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

# user selects CSV file
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


data = data['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
data['Potential (V)'] = data['Adjusted Potential (V)']
data['Capacity (mAh/g)'] = data['Adjusted Relative Capacity (mAh/g)']
data = data['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']

# stating input + output columns
input_features = ['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
output_features = ['Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Cycle']

# seperating data
data_input = data[input_features]
data_output = data[output_features]

scaler_input = MinMaxScaler()
data_input_scaled = scaler_input.fit_transform(data_input)

scaler_output = MinMaxScaler()
data_output_scaled = scaler_output.fit_transform(data_output)

# Combine input and output data WHY???
scaled_data = pd.DataFrame(np.hstack([data_input_scaled, data_output_scaled]),columns=input_features + output_features)


# ======================
# 2. Sequence Creation
# ======================
def create_sequences(data, sequence_length=10):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-3])  # Use all input features (last 3 columns are output)
        y.append(data[i + sequence_length, -3:])  # Predict output features
    return np.array(X), np.array(y)

# Create sequences for training
sequence_length = 10  # You can change this based on your data and how far back you want to look
X, y = create_sequences(scaled_data.values, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)



# ======================
# 3. Build Model
# ======================
# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We only care about the output of the last time step
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out



# ======================
# 4. Train Model
# ======================
# Hyperparameters
input_size = 7  # Number of input features at each time step
hidden_size = 64  # Number of hidden units in LSTM
output_size = 3  # Number of output features
num_epochs = 50  # Number of training epochs
batch_size = 32  # Batch size for training
learning_rate = 0.001  # Learning rate

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
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
# 5. evaluate model
# ======================
# Switch to evaluation mode
model.eval()

# Make predictions on the training data
with torch.no_grad():
    predictions = model(X_tensor)

# Rescale the predictions to original scale
predictions_rescaled = scaler_output.inverse_transform(predictions.numpy())

# Plot actual vs predicted values for one feature (e.g., Zre)
plt.plot(y_tensor[:, 0].numpy(), label="Actual Zre")
plt.plot(predictions_rescaled[:, 0], label="Predicted Zre")
plt.legend()
plt.show()

