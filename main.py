import pandas as pd
import numpy as np          # for data handling
import matplotlib.pyplot as plt   # for plotting

filePath = r'C:\Users\Pengu\OneDrive\Documents\S21FarApart.csv'

try:
    df = pd.read_csv(filePath)
except FileNotFoundError:
    print(f"Error: The file '{filePath}' was not found. Please check the file path.")
    # Exit or handle the error appropriately
    exit()

print("Data successfully loaded:")

dfClean = df.dropna().reset_index(drop=True)
# Find resonant peak
f0 = 8e9
S21_at_f0 = np.interp(f0, dfClean['Frequency'], dfClean['Mag'])
half_power = S21_at_f0 + 3


# Function to interpolate the -3 dB crossing
def interpolate(f1, f2, y1, y2, target):
    return f1 + (target - y1) * (f2 - f1) / (y2 - y1)

# Left -3 dB point
left_data = dfClean[dfClean['Frequency'] < f0]
for i in range(len(left_data)-1, 0, -1):
    if left_data['Mag'].iloc[i-1] < half_power <= left_data['Mag'].iloc[i]:
        f1 = interpolate(left_data['Frequency'].iloc[i-1],
                         left_data['Frequency'].iloc[i],
                         left_data['Mag'].iloc[i-1],
                         left_data['Mag'].iloc[i],
                         half_power)
        break

# Right -3 dB point
right_data = dfClean[dfClean['Frequency'] > f0]
for i in range(len(right_data)-1):
    if right_data['Mag'].iloc[i] >= half_power > right_data['Mag'].iloc[i+1]:
        f2 = interpolate(right_data['Frequency'].iloc[i],
                         right_data['Frequency'].iloc[i+1],
                         right_data['Mag'].iloc[i],
                         right_data['Mag'].iloc[i+1],
                         half_power)
        break

# Compute Q
bandwidth = f2 - f1
Q = f0 / bandwidth

# Convert to GHz
f0G = f0/(1e9)
bandwidthG = bandwidth/(1e9)

print(f"Resonant frequency f0 = {f0G:.2f} GHz")
print(f"Bandwidth Δf = {bandwidthG:.2f} GHz")
print(f"Q factor = {Q:.2f}")
