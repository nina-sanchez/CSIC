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
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # ======================
# # 1. Load + Clean Data
# # ======================
# Tk().withdraw()
# file_path = askopenfilename(title="Select a .csv file", filetypes=[("CSV files", "*.csv")])
# if not file_path:
#     print("No file selected.")
#     exit()

# try:
#     data = pd.read_csv(file_path)
#     print(f"File loaded: {file_path}")
# except Exception as e:
#     print(f"Error loading file: {e}")
#     exit()

# # Clean + rename columns
# data = data[['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']]
# data = data.dropna()
# data['Potential (V)'] = data['Adjusted Potential (V)']
# data['Capacity (mAh/g)'] = data['Adjusted Relative Capacity (mAh/g)']
# data = data[['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle',
#              'Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)']]

# # Inputs and targets
# input_cols = ['Potential (V)', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
# target_cols = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)']

# # ======================
# # 2. Train/Test Split
# # ======================
# target_cycle = 6
# train_data = data[data['Cycle'] < target_cycle]
# test_data = data[data['Cycle'] == target_cycle]

# # ======================
# # 3. Normalize
# # ======================
# scaler_input = MinMaxScaler()
# scaler_target = MinMaxScaler()

# train_input_scaled = scaler_input.fit_transform(train_data[input_cols])
# train_target_scaled = scaler_target.fit_transform(train_data[target_cols])

# test_input_scaled = scaler_input.transform(test_data[input_cols])
# test_target_scaled = scaler_target.transform(test_data[target_cols])

# # ======================
# # 4. Create Sequences
# # ======================
# def create_sequences(inputs, targets, seq_len=10):
#     X, y = [], []
#     for i in range(len(inputs) - seq_len):
#         X.append(inputs[i:i + seq_len])
#         y.append(targets[i + seq_len])
#     return np.array(X), np.array(y)

# sequence_length = 10
# X_train, y_train = create_sequences(train_input_scaled, train_target_scaled, sequence_length)
# X_test, y_test = create_sequences(test_input_scaled, test_target_scaled, sequence_length)

# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# # ======================
# # 5. LSTM Model
# # ======================
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# model = LSTMModel(input_size=X_train.shape[2], hidden_size=64, output_size=2)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # ======================
# # 6. Train Model
# # ======================
# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     output = model(X_train_tensor)
#     loss = criterion(output, y_train_tensor)
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# # ======================
# # 7. Predict + Plot
# # ======================
# model.eval()
# with torch.no_grad():
#     preds = model(X_test_tensor).numpy()
#     preds_rescaled = scaler_target.inverse_transform(preds)

# # Actual test target for plotting
# actual = scaler_target.inverse_transform(y_test_tensor.numpy())

# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(actual[:, 1], actual[:, 0], label='Actual', color='blue')
# plt.plot(preds_rescaled[:, 1], preds_rescaled[:, 0], label='Predicted', color='red', linestyle='--')

# plt.xlabel("Capacidad Específica (mAh/g)")
# plt.ylabel("Potencial (V)")
# plt.title(f"Actual vs Predicted – Ciclo {target_cycle}")
# plt.legend()
# plt.grid()
# plt.show()


# WORKIMNG


# feature importance, randomforestregressor
# tells us the importance of using certain columns
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# 1. Load Data
# ======================
Tk().withdraw()
file_path = askopenfilename(title="Select a .csv file", filetypes=[("CSV files", "*.csv")])
if not file_path:
    print("No file selected.")
    exit()

try:
    data = pd.read_csv(file_path)
    print(f"File loaded: {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# ======================
# 2. Clean and Prepare
# ======================
required_cols = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle', '']
data = data[required_cols].dropna()

# features and targets
input_cols = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
target_cols = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)']

# rename for clarity
data = data.rename(columns={
    'Adjusted Potential (V)': 'Target_Potential',
    'Adjusted Relative Capacity (mAh/g)': 'Target_Capacity'
})
input_cols = ['Target_Potential', 'Target_Capacity', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
target_cols = ['Target_Potential', 'Target_Capacity']

# ======================
# 3. Train/Test Split
# ======================
target_cycle = 17
train_data = data[data['Cycle'] < target_cycle]
test_data = data[data['Cycle'] == target_cycle]

# inputs and targets
X_train = train_data[input_cols].copy()
X_test = test_data[input_cols].copy()
y_train = train_data[target_cols].copy()
y_test = test_data[target_cols].copy()

# ======================
# 4. Normalize
# ======================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ======================
# 5. Train Random Forest
# ======================
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train_scaled, y_train_scaled)

# chat added
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_scaled)  # y_train_scaled shape: (n_samples, 2)

# Predict on test set
y_pred_scaled = rf.predict(X_test_scaled)  # shape: (n_samples, 2)

# Inverse transform predictions and actual scaled targets back to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)   # shape: (n_samples, 2)
y_test_orig = scaler_y.inverse_transform(y_test_scaled)  # shape: (n_samples, 2)

# Calculate MSE and R^2 for both output columns together
mse = mean_squared_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# ======================
# 6. Predict + Evaluate
# ======================
y_pred_scaled = rf.predict(X_test_scaled)
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
print(f"Random Forest Test MSE: {mse:.6f}")

# inverse transform to real values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test_scaled)

# ======================
# 7. Plot Results
# ======================
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(y_test_real[:, 0], label='True Potential (V)')
# plt.plot(y_pred[:, 0], label='Predicted Potential (V)')
# plt.title('Adjusted Potential (V)')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(y_test_real[:, 1], label='True Capacity (mAh/g)')
# plt.plot(y_pred[:, 1], label='Predicted Capacity (mAh/g)')
# plt.title('Adjusted Capacity (mAh/g)')
# plt.legend()

# plt.tight_layout()
# plt.show()




# THIS WORKS
# true_capacity = y_test_real[:, 1]
# true_potential = y_test_real[:, 0]
# pred_capacity = y_pred[:, 1]
# pred_potential = y_pred[:, 0]

# plt.figure(figsize=(8, 6))
# plt.plot(true_capacity, true_potential, label='True Cycle Data', linewidth=2)
# plt.plot(pred_capacity, pred_potential, label='Predicted Cycle Data', linestyle='--', linewidth=2)
# plt.xlabel('Adjusted Relative Capacity (mAh/g)')
# plt.ylabel('Adjusted Potential (V)')
# plt.title(f'Cycle {target_cycle} - True vs Predicted')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




plt.figure(figsize=(10, 8))

for cycle in range(1, target_cycle + 1):  # cycles 1 to 17 inclusive
    # Select data for the current cycle
    cycle_data = data[data['Cycle'] == cycle]
    
    # Prepare inputs and targets for this cycle (must be consistent with your scaling)
    X_cycle = cycle_data[input_cols]
    y_cycle = cycle_data[target_cols]
    
    # Scale inputs and targets
    X_cycle_scaled = scaler_X.transform(X_cycle)
    y_cycle_scaled = scaler_y.transform(y_cycle)
    
    # Predict and inverse transform
    y_cycle_pred_scaled = rf.predict(X_cycle_scaled)
    y_cycle_pred = scaler_y.inverse_transform(y_cycle_pred_scaled)
    y_cycle_true = scaler_y.inverse_transform(y_cycle_scaled)
    
    # Extract true and predicted capacity and potential
    true_capacity = y_cycle_true[:, 1]
    true_potential = y_cycle_true[:, 0]
    pred_capacity = y_cycle_pred[:, 1]
    pred_potential = y_cycle_pred[:, 0]
    
    # Plot true data (solid) and predicted data (dashed) for this cycle
    plt.plot(true_capacity, true_potential, label=f'True Cycle {cycle}', linewidth=1.5)
    plt.plot(pred_capacity, pred_potential, linestyle='--', label=f'Predicted Cycle {cycle}', linewidth=1.5)

plt.xlabel('Adjusted Relative Capacity (mAh/g)')
plt.ylabel('Adjusted Potential (V)')
plt.title('True vs Predicted Data for Cycles 1 to 17')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # place legend outside plot
plt.grid(True)
plt.tight_layout()
plt.show()





# After training the Random Forest
importances = rf.feature_importances_
feature_names = input_cols  # same order as your training features

# Print feature importances
print("Feature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Optional: Plot feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()  # highest importance on top
plt.tight_layout()
plt.show()
