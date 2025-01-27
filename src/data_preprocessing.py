import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler  # More efficient scaling

# Load only necessary columns and specify dtypes for faster loading
columns = ['Datetime', 'Temperature', 'Humidity', 'WindSpeed',
           'GeneralDiffuseFlows', 'DiffuseFlows',
           'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']
# Use float32 instead of float64
dtype_dict = {col: 'float32' for col in columns[1:]}
dtype_dict['Datetime'] = str

data = pd.read_csv(
    'Hands-on-Activity-Forecasting-Power-Consumption/data/powerconsumption.csv',
    usecols=columns,
    dtype=dtype_dict,
    parse_dates=['Datetime']
)

# More efficient data handling
data.sort_values('Datetime', inplace=True)
data.ffill(inplace=True)
data.bfill(inplace=True)

# Efficient splitting without copying
total_samples = len(data)
test_size = int(total_samples * 0.2)
val_size = int(total_samples * 0.1)

train_idx = slice(None, -test_size-val_size)
val_idx = slice(-test_size-val_size, -test_size)
test_idx = slice(-test_size, None)


def tokenize_data(df, scaler=None):
    features = df[[col for col in columns if col !=
                   'Datetime']].values.astype(np.float32)

    if scaler is None:
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
        scaling_params = {
            'min_vals': scaler.data_min_.tolist(),  # Changed from min_ to data_min_
            'max_vals': scaler.data_max_.tolist()   # Changed from max_ to data_max_
        }
        return normalized_features, scaling_params, scaler
    else:
        normalized_features = scaler.transform(features)
        return normalized_features, None, None


# Process datasets
train_tokens, scaling_params, scaler = tokenize_data(data.iloc[train_idx])
val_tokens, _, _ = tokenize_data(data.iloc[val_idx], scaler)
test_tokens, _, _ = tokenize_data(data.iloc[test_idx], scaler)

# Save processed data
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/train_data.npy', train_tokens)
np.save('data/processed/val_data.npy', val_tokens)
np.save('data/processed/test_data.npy', test_tokens)

with open('data/processed/scaling_params.json', 'w') as f:
    json.dump(scaling_params, f)

# Debug prints
print("\nDataset shapes:")
print(f"Total data: {data.shape}")
print(f"Train data: {data.iloc[train_idx].shape}")
print(f"Validation data: {data.iloc[val_idx].shape}")
print(f"Test data: {data.iloc[test_idx].shape}")

print("\nSample of normalized features (first 5 rows of train set):")
print(train_tokens[:5])

print("\nScaling parameters:")
print("Min values:", scaling_params['min_vals'])
print("Max values:", scaling_params['max_vals'])

# Check for any remaining NaN values
print("\nNaN check:")
print(data.isna().sum())
