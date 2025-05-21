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
required_cols = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
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

# debug outputs
print("X_test indices:", X_test.index.tolist())
print("y_test indices:", y_test.index.tolist())


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

print("\nSample Predictions vs Actuals:")
for i in range(min(5, len(y_pred))):
    print(f"Pred: Potential={y_pred[i][0]:.2f}, Capacity={y_pred[i][1]:.2f} | "
          f"Actual: Potential={y_test_orig[i][0]:.2f}, Capacity={y_test_orig[i][1]:.2f}")
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
# 7a. Plot Results
# ======================
# results_df = pd.DataFrame({
#     'Cycle': test_data['Cycle'].values,
#     'Frequency (Hz)': test_data['Frequency (Hz)'].values,
#     'Actual Potential (V)': y_test_orig[:, 0],
#     'Predicted Potential (V)': y_pred[:, 0],
#     'Actual Capacity (mAh/g)': y_test_orig[:, 1],
#     'Predicted Capacity (mAh/g)': y_pred[:, 1],
# })
results_df = pd.DataFrame({
    'Cycle': test_data['Cycle'].values,
    'Frequency (Hz)': test_data['Frequency (Hz)'].values,
    'Actual Potential (V)': test_data['Target_Potential'].values,
    'Predicted Potential (V)': y_pred[:, 0],
    'Actual Capacity (mAh/g)': test_data['Target_Capacity'].values,
    'Predicted Capacity (mAh/g)': y_pred[:, 1],
})


# Sort by frequency (optional, useful for EIS plots)
results_df = results_df.sort_values(by='Frequency (Hz)', ascending=False)

# Display the table
print(results_df.to_string(index=False))
results_df.to_csv("cycle_17_predictions.csv", index=False)

# ======================
# 7b. Plot Results
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
