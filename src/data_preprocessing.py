import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv(
    'Hands-on-Activity-Forecasting-Power-Consumption/data/powerconsumption.csv',
    parse_dates=['Datetime']
)

# Handle missing values using forward fill (appropriate for time series)
data.ffill(inplace=True)
# If any remaining NaN at the start, backward fill
data.bfill(inplace=True)

# Ensure data is sorted by datetime
data = data.sort_values('Datetime')

# Split the data into training, validation, and test sets
# Using a rolling window approach for time series
# Last 20% for test, previous 10% for validation, rest for training
total_samples = len(data)
test_size = int(total_samples * 0.2)
val_size = int(total_samples * 0.1)

train_data = data[:-test_size-val_size]
val_data = data[-test_size-val_size:-test_size]
test_data = data[-test_size:]


def tokenize_data(df):
    # Extract only the raw numerical features
    features = df[[
        'Temperature', 'Humidity', 'WindSpeed',
        'GeneralDiffuseFlows', 'DiffuseFlows',
        'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'
    ]].values

    # Normalize the features to help with model training
    # Using min-max scaling
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    normalized_features = (features - min_vals) / (max_vals - min_vals)

    return normalized_features


# Create tokens for each dataset
train_tokens = tokenize_data(train_data)
val_tokens = tokenize_data(val_data)
test_tokens = tokenize_data(test_data)

# Save the scaling parameters for later use
feature_columns = ['Temperature', 'Humidity', 'WindSpeed',
                   'GeneralDiffuseFlows', 'DiffuseFlows',
                   'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']

scaling_params = {
    'min_vals': dict(zip(feature_columns, train_tokens.min(axis=0))),
    'max_vals': dict(zip(feature_columns, train_tokens.max(axis=0)))
}

# Debug prints
print("\nDataset shapes:")
print(f"Total data: {data.shape}")
print(f"Train data: {train_data.shape}")
print(f"Validation data: {val_data.shape}")
print(f"Test data: {test_data.shape}")

print("\nSample of normalized features (first 5 rows of train set):")
print(train_tokens[:5])

print("\nScaling parameters:")
print("Min values:", scaling_params['min_vals'])
print("Max values:", scaling_params['max_vals'])

# Check for any remaining NaN values
print("\nNaN check:")
print(data.isna().sum())
