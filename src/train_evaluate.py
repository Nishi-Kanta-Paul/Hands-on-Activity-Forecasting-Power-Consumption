import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from vanilla_transformer import create_transformer_model, prepare_data, train_transformer, evaluate_transformer
from patchtst import create_patchtst_model, prepare_data_patchtst, train_patchtst, evaluate_patchtst
import json
import os
from datetime import datetime
import pandas as pd


class ModelTrainer:
    def __init__(self, train_data, val_data, test_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Convert to float32 for faster training
        self.train_data = torch.FloatTensor(train_data).to(device)
        self.val_data = torch.FloatTensor(val_data).to(device)
        self.test_data = torch.FloatTensor(test_data).to(device)
        self.device = device
        self.metrics_history = {}

    def prepare_dataloaders(self, model_type, seq_length, batch_size, pred_length=1):
        # Prepare data in parallel
        if model_type == 'transformer':
            prepare_fn = prepare_data
        else:
            prepare_fn = prepare_data_patchtst

        # Use smaller subset of data for faster training
        # Use only half of training data
        train_size = len(self.train_data) // 2
        train_sequences, train_targets = prepare_fn(
            self.train_data[:train_size], seq_length, pred_length)

        val_size = len(self.val_data) // 2  # Use only half of validation data
        val_sequences, val_targets = prepare_fn(
            self.val_data[:val_size], seq_length, pred_length)

        test_size = len(self.test_data) // 2  # Use only half of test data
        test_sequences, test_targets = prepare_fn(
            self.test_data[:test_size], seq_length, pred_length)

        # Optimize DataLoader settings
        train_loader = DataLoader(
            TensorDataset(train_sequences, train_targets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,  # Reduced to 1 for less overhead
            pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(val_sequences, val_targets),
            batch_size=batch_size * 4,  # Even larger batches for validation
            num_workers=1,
            pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(test_sequences, test_targets),
            batch_size=batch_size * 4,  # Even larger batches for testing
            num_workers=1,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def train_model(self, model_type, config, num_epochs=100, patience=10):
        # Prepare dataloaders first
        train_loader, val_loader, test_loader = self.prepare_dataloaders(
            model_type,
            config['seq_length'],
            config['batch_size'],
            config.get('pred_length', 1)
        )

        # Use gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        model = (create_transformer_model(config) if model_type == 'transformer'
                 else create_patchtst_model(config))
        model = model.to(self.device)

        # Enable automatic mixed precision
        model = model.half() if self.device == 'cuda' else model

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()

        # Use learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Train
            if model_type == 'transformer':
                train_loss = train_transformer(model, train_loader,
                                               criterion, optimizer, self.device)
            else:
                train_loss = train_patchtst(model, train_loader,
                                            criterion, optimizer, self.device)
            train_losses.append(train_loss)

            # Validate
            if model_type == 'transformer':
                val_loss = evaluate_transformer(
                    model, val_loader, criterion, self.device)
            else:
                val_loss = evaluate_patchtst(
                    model, val_loader, criterion, self.device)
            val_losses.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(model, model_type, config)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break

        # Save training history
        self.metrics_history[model_type] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        return model, train_losses, val_losses

    def evaluate_model(self, model, model_type, config):
        """Evaluate model and return metrics"""
        _, _, test_loader = self.prepare_dataloaders(
            model_type, config['seq_length'], config['batch_size'])

        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                if model_type == 'transformer':
                    output = model(data, None)  # None for mask
                    pred = output[:, -1, :]  # Get last timestep predictions
                else:
                    output = model(data)
                    pred = output

                predictions.extend(pred.cpu().numpy())
                actuals.extend(target.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)

        # Convert numpy types to Python native types
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse)
        }

        return metrics, predictions, actuals

    def save_model(self, model, model_type, config):
        """Save model and configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'models/{model_type}_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)

        # Save model state
        torch.save(model.state_dict(), f'{save_dir}/model.pth')

        # Save configuration
        with open(f'{save_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def plot_training_history(self, model_type):
        """Plot training and validation losses"""
        history = self.metrics_history[model_type]
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_losses'], label='Training Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title(f'{model_type} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'plots/{model_type}_training_history.png')
        plt.close()

    def save_results(self, transformer_model, patchtst_model, config, test_loader):
        """
        Save model predictions and actual values for analysis
        """
        # Create results directory if it doesn't exist
        os.makedirs('../data/results', exist_ok=True)

        transformer_model.eval()
        patchtst_model.eval()
        all_transformer_preds = []
        all_patchtst_preds = []
        all_actuals = []

        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                # Move data to device
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)

                # Get predictions from both models
                transformer_output = transformer_model(
                    batch_data, None)  # None for mask
                # Get last timestep
                transformer_pred = transformer_output[:, -1, :]

                patchtst_output = patchtst_model(batch_data)
                patchtst_pred = patchtst_output

                # Store predictions and actuals
                all_transformer_preds.append(transformer_pred.cpu().numpy())
                all_patchtst_preds.append(patchtst_pred.cpu().numpy())
                all_actuals.append(batch_target.cpu().numpy())

        # Concatenate all batches
        transformer_predictions = np.concatenate(all_transformer_preds, axis=0)
        patchtst_predictions = np.concatenate(all_patchtst_preds, axis=0)
        actual_values = np.concatenate(all_actuals, axis=0)

        # Save the arrays
        np.save('../data/results/transformer_predictions.npy',
                transformer_predictions)
        np.save('../data/results/patchtst_predictions.npy', patchtst_predictions)
        np.save('../data/results/actual_values.npy', actual_values)

        # Save configuration and metrics
        results_info = {
            'config': config,
            'data_shape': {
                'transformer_predictions': transformer_predictions.shape,
                'patchtst_predictions': patchtst_predictions.shape,
                'actual_values': actual_values.shape
            },
            'timestamp': str(pd.Timestamp.now())
        }

        with open('../data/results/results_info.json', 'w') as f:
            json.dump(results_info, f, indent=4)

        print("\nResults saved successfully!")
        print(f"Transformer predictions shape: {
              transformer_predictions.shape}")
        print(f"PatchTST predictions shape: {patchtst_predictions.shape}")
        print(f"Actual values shape: {actual_values.shape}")


def main():
    # Load preprocessed data
    train_data = np.load('data/processed/train_data.npy')
    val_data = np.load('data/processed/val_data.npy')
    test_data = np.load('data/processed/test_data.npy')

    # Initialize trainer
    trainer = ModelTrainer(train_data, val_data, test_data)

    # Minimal configurations for fastest training
    transformer_config = {
        'input_dim': 8,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'output_dim': 3,
        'dropout': 0.1,
        'seq_length': 24,
        'batch_size': 128,
        'learning_rate': 0.001
    }

    patchtst_config = {
        'input_dim': 8,
        'output_dim': 3,
        'patch_len': 6,
        'stride': 6,
        'num_patches': 4,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'dropout': 0.1,
        'seq_length': 24,
        'batch_size': 128,
        'learning_rate': 0.001
    }

    # Minimal training time
    num_epochs = 10
    patience = 3

    # Train models
    print("Training Vanilla Transformer...")
    transformer_model, transformer_train_losses, transformer_val_losses = trainer.train_model(
        'transformer', transformer_config, num_epochs=num_epochs, patience=patience)

    print("\nTraining PatchTST...")
    patchtst_model, patchtst_train_losses, patchtst_val_losses = trainer.train_model(
        'patchtst', patchtst_config, num_epochs=num_epochs, patience=patience)

    # Create test loader for saving results
    _, _, test_loader = trainer.prepare_dataloaders(
        'transformer',  # Use transformer config for test loader
        transformer_config['seq_length'],
        transformer_config['batch_size']
    )

    # Save results
    trainer.save_results(
        transformer_model,
        patchtst_model,
        {
            'transformer_config': transformer_config,
            'patchtst_config': patchtst_config
        },
        test_loader
    )

    # Print metrics
    print("\nTransformer Metrics:")
    transformer_metrics, _, _ = trainer.evaluate_model(
        transformer_model, 'transformer', transformer_config)
    print(json.dumps(transformer_metrics, indent=4))

    print("\nPatchTST Metrics:")
    patchtst_metrics, _, _ = trainer.evaluate_model(
        patchtst_model, 'patchtst', patchtst_config)
    print(json.dumps(patchtst_metrics, indent=4))


if __name__ == "__main__":
    main()
