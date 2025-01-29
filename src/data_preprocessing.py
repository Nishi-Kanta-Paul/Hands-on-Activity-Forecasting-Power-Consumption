import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler  # More efficient scaling
import pickle

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


class DataPreprocessor:
    def __init__(self):
        # Create necessary directories
        os.makedirs('data/processed', exist_ok=True)

        # Initialize data attributes
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_columns = None
        self.scaler = None

        # Set data paths
        self.data_path = 'Hands-on-Activity-Forecasting-Power-Consumption/data/powerconsumption.csv'
        self.processed_dir = 'data/processed'

    def preprocess(self):
        """Main preprocessing function"""
        # Load data
        columns = ['Datetime', 'Temperature', 'Humidity', 'WindSpeed',
                   'GeneralDiffuseFlows', 'DiffuseFlows',
                   'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']

        dtype_dict = {col: 'float32' for col in columns[1:]}
        dtype_dict['Datetime'] = str

        try:
            data = pd.read_csv(
                self.data_path,
                usecols=columns,
                dtype=dtype_dict,
                parse_dates=['Datetime']
            )
        except FileNotFoundError:
            # Try alternative path
            alt_path = 'data/powerconsumption.csv'
            print(f"Trying alternative path: {alt_path}")
            data = pd.read_csv(
                alt_path,
                usecols=columns,
                dtype=dtype_dict,
                parse_dates=['Datetime']
            )

        print(f"Successfully loaded data with shape: {data.shape}")

        # Store feature columns
        self.feature_columns = columns[1:]  # All columns except Datetime

        # Prepare data
        data.sort_values('Datetime', inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        # Split data
        total_samples = len(data)
        test_size = int(total_samples * 0.2)
        val_size = int(total_samples * 0.1)

        train_idx = slice(None, -test_size-val_size)
        val_idx = slice(-test_size-val_size, -test_size)
        test_idx = slice(-test_size, None)

        # Process and store datasets
        self.train_data, scaling_params, self.scaler = self.tokenize_data(
            data.iloc[train_idx])
        self.val_data, _, _ = self.tokenize_data(
            data.iloc[val_idx], self.scaler)
        self.test_data, _, _ = self.tokenize_data(
            data.iloc[test_idx], self.scaler)

        # Save scaling parameters
        with open('data/processed/scaling_params.json', 'w') as f:
            json.dump(scaling_params, f)

        # Save the preprocessed data
        self.save_preprocessed_data()

    def tokenize_data(self, df, scaler=None):
        """Tokenize and normalize the data"""
        features = df[[col for col in self.feature_columns if col !=
                       'Datetime']].values.astype(np.float32)

        if scaler is None:
            scaler = MinMaxScaler()
            normalized_features = scaler.fit_transform(features)
            scaling_params = {
                'min_vals': scaler.data_min_.tolist(),
                'max_vals': scaler.data_max_.tolist()
            }
            return normalized_features, scaling_params, scaler
        else:
            normalized_features = scaler.transform(features)
            return normalized_features, None, None

    def save_preprocessed_data(self):
        """Save preprocessed data to files"""
        # Save train, validation, and test data
        np.save('data/processed/train_data.npy', self.train_data)
        np.save('data/processed/val_data.npy', self.val_data)
        np.save('data/processed/test_data.npy', self.test_data)

        # Save preprocessing info
        preprocessing_info = {
            'train_shape': self.train_data.shape,
            'val_shape': self.val_data.shape,
            'test_shape': self.test_data.shape,
            'features': self.feature_columns,
            'timestamp': str(pd.Timestamp.now())
        }

        with open('data/processed/preprocessing_info.json', 'w') as f:
            json.dump(preprocessing_info, f, indent=4)

        print("\nPreprocessed data saved successfully!")
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Validation data shape: {self.val_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")


def main():
    preprocessor = DataPreprocessor()
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
