import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt


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


# --- DATA CLEANING + ORGANIZING ---
# limit rows that it gets
# data = data.head(50000)

data = data[data["ActionId"] != 41] # remove 41
# data = data[data["ActionId"] != 8] # remove charge + discharge data
data["Elapsed Time (h)"] = data["Elapsed Time (s)"] / 3600
data["Type"] = np.where(data["Current (A)"] > 0, "1", "0") # identify C + D
data['Cycle'] = np.nan # adding a cycle column
data['Capacity (mAh/g)'] = np.nan # adding a mAh/g column
data['Relative Time (h)'] = np.nan # relative time 
data['Relative Capacity (mAh/g)'] = np.nan # relative capacity in terms of relative time

# new columns for adjusted potential and relative capacity
data["Adjusted Potential (V)"] = np.nan
data["Adjusted Relative Capacity (mAh/g)"] = np.nan

# print(data.columns)
    
# create lists for C+D segments
charge_segment = []
discharge_segment = []

# organizing segements
all_segments = data["Segment"]
current_segment = [all_segments.iloc[0]]

# labeling as either charge or discharge
charge_type = "Charge" if data[data["Segment"] == all_segments.iloc[0]]["Current (A)"].mean() > 0 else "Discharge"

# seperating charges and discharges
for seg in all_segments.iloc[1:]:
    segment_type = "Charge" if data[data["Segment"] == seg]["Current (A)"].mean() > 0 else "Discharge"
    if segment_type == charge_type:
        current_segment.append(seg)
    else:
        if charge_type == "Charge":
            charge_segment.append(current_segment)
        else:
            discharge_segment.append(current_segment)
        current_segment = [seg]
        charge_type = segment_type

if charge_type == "Charge":
    charge_segment.append(current_segment)
else:
    discharge_segment.append(current_segment)

# adding # of cycle
num_cycles = min(len(charge_segment), len(discharge_segment))
for i in range(num_cycles):
    cycle = charge_segment[i] + discharge_segment[i]
    data.loc[data['Segment'].isin(cycle), 'Cycle'] = i + 1
data['Cycle'] = data['Cycle'].astype('Int64')  # remove decimal


# adding capacity --> not correct, follow relative capacity
MASA_NMC = 0.0044
data['Capacity (mAh/g)'] = (data["Elapsed Time (h)"] * (data["Current (A)"].abs() * 1000)) / MASA_NMC


# calculating relative capacity + relative time
capacity_all = pd.Series(index=data.index, dtype=float)
relative_time_all = pd.Series(index=data.index, dtype=float)

# loop over all charge/discharge blocks
for block in charge_segment + discharge_segment:
    # mask for this block --> all rows in block
    mask = data['Segment'].isin(block)
    block_data = data.loc[mask]

    # relative time per block
    time_h = block_data['Elapsed Time (h)']
    current_mA = block_data['Current (A)'].abs() * 1000  # convert A to mA
    relative_time = time_h - time_h.iloc[0]

    # calculate delta time and cumulative capacity
    delta_t = np.diff(relative_time, prepend=relative_time.iloc[0])
    capacity = np.cumsum(delta_t * current_mA) / MASA_NMC

    # assign back to the full dataframe
    capacity_all.loc[mask] = capacity
    relative_time_all.loc[mask] = relative_time

# save to data
data["Relative Time (h)"] = relative_time_all
data["Relative Capacity (mAh/g)"] = capacity_all



# THIS IS FOR DEBUGGING --> NEED TO FILTER BEFORE TO LEAVE AS '0'
# bad_freqs = data[data["Frequency (Hz)"] <= 0]
# if not bad_freqs.empty:
#     print("found invalid frequencies (zero or negative):")
#     print(bad_freqs[["Frequency (Hz)"]])

# log scale the frequency
# data['Frequency (Hz)'] = np.log10(data['Frequency (Hz)'])
# data['Frequency (Hz)'] = np.where(
#     data['Frequency (Hz)'] > 0,
#     np.log10(data['Frequency (Hz)']),
#     0.0 # will replace with 0.0 if empty
# )


# First, create a copy of the column
# freq_values = data['Frequency (Hz)'].copy()

# # Apply log10 only to positive values
# log_freq = np.where(freq_values > 0, np.log10(freq_values), 0.0)

# # Assign it back
# data['Frequency (Hz)'] = log_freq


# # adding last C or D row to first imp. row and rest blank
# # Identify rows with actionID == 21
# impedance_mask = data["ActionId"] == 21

# # find index positions of all impedance rows
# impedance_indices = data.index[impedance_mask]

# # track last valid row (with ActionID == 8)
# last_valid_idx = None

# # loop through the impedance indices to identify segment starts
# for i, idx in enumerate(impedance_indices):
#     if i == 0 or impedance_indices[i-1] != idx - 1:
#         # this is the first row of a new segment
#         # look back for last row with ActionID == 8
#         # last_valid_idx = data.loc[:idx-1][data["ActionId"] == 8].last_valid_index()
#         subset = data.loc[:idx-1]
#         last_valid_idx = subset[subset["ActionId"] == 8].last_valid_index()

        
#         if last_valid_idx is not None:
#             # copy potential and capacity from the previous valid row
#             data.at[idx, "Adjusted Potential (V)"] = data.at[last_valid_idx, "Potential (V)"]
#             data.at[idx, "Adjusted Relative Capacity (mAh/g)"] = data.at[last_valid_idx, "Relative Capacity (mAh/g)"]

# Initialize placeholders
last_valid_potential = None
last_valid_capacity = None

# Loop through each row
for idx, row in data.iterrows():
    if row["ActionId"] == 8:
        # Update the "last known" potential and capacity
        last_valid_potential = row["Potential (V)"]
        last_valid_capacity = row["Relative Capacity (mAh/g)"]
        
    elif row["ActionId"] == 21:
        # Replace with most recent valid values
        data.at[idx, "Adjusted Potential (V)"] = last_valid_potential
        data.at[idx, "Adjusted Relative Capacity (mAh/g)"] = last_valid_capacity



# ROUNDING FINAL VALUES
data['Adjusted Potential (V)'] = data['Adjusted Potential (V)'].round(2)
data['Adjusted Relative Capacity (mAh/g)'] = data['Adjusted Relative Capacity (mAh/g)'].round(2)
data['Frequency (Hz)'] = data['Frequency (Hz)'].round(2)
data['Zre (ohms)'] = data['Zre (ohms)'].round(2)
data['Zim (ohms)'] = data['Zim (ohms)'].round(2)

# --------------------------- Output Data ----------------------------- #
# select the columns wanted
# data_columns = ['Segment', 'Elapsed Time (h)', 'Relative Time (h)', 'Potential (V)', 'Current (A)', 'Relative Capacity', 'Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle', 'ActionId']
# data_columns = ['Segment', 'Elapsed Time (s)', 'Elapsed Time (h)', 'Relative Time (h)', 'Relative Capacity', 'Capacity (mAh/g)', 'Type', 'Cycle']
# data_columns = ['Potential (V)', 'Adjusted Potential (V)', 'Relative Capacity (mAh/g)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle', 'ActionId']


data_columns = ['Adjusted Potential (V)', 'Adjusted Relative Capacity (mAh/g)', 'Frequency (Hz)', 'Zre (ohms)', 'Zim (ohms)', 'Type', 'Cycle']
filtered_data = data[data_columns]

filtered_data = data[data["ActionId"] == 21][data_columns]

# save to CSV
# filtered_data.to_csv("full_segment_data.csv", index=False)
filtered_data.to_csv("final-data-1302.csv", index=False)
print("Full data saved as 'final-data-1302.csv' with", len(filtered_data), "rows")




# pseudocode for AI
# read in file

# select columns that will be fed: frequency, zre, zim, potential, capacity
# output - potential, capacity

# plot results generated vs real results
    
