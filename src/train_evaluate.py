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

        train_sequences, train_targets = prepare_fn(
            self.train_data, seq_length, pred_length)
        val_sequences, val_targets = prepare_fn(
            self.val_data, seq_length, pred_length)
        test_sequences, test_targets = prepare_fn(
            self.test_data, seq_length, pred_length)

        # Optimize DataLoader settings
        train_loader = DataLoader(
            TensorDataset(train_sequences, train_targets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced from 4 for faster startup
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        )
        val_loader = DataLoader(
            TensorDataset(val_sequences, val_targets),
            batch_size=batch_size * 2,  # Larger batches for validation
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            TensorDataset(test_sequences, test_targets),
            batch_size=batch_size * 2,  # Larger batches for testing
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
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
            train_loss = train_transformer(model, train_loader,
                                           criterion, optimizer, self.device)
            train_losses.append(train_loss)

            # Validate
            val_loss = evaluate_transformer(
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
        """Evaluate model performance on test set"""
        _, _, test_loader = self.prepare_dataloaders(
            model_type,
            config['seq_length'],
            config['batch_size'],
            config.get('pred_length', 1)
        )

        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                if model_type == 'transformer':
                    mask = torch.triu(torch.ones(
                        data.size(1), data.size(1)), diagonal=1).bool()
                    mask = mask.to(self.device)
                    output = model(data, mask)
                    output = output[:, -1, :]  # Get last timestep predictions
                else:  # PatchTST
                    output = model(data)

                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100
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


def main():
    # Load preprocessed data
    train_data = np.load('data/processed/train_data.npy')
    val_data = np.load('data/processed/val_data.npy')
    test_data = np.load('data/processed/test_data.npy')

    # Initialize trainer
    trainer = ModelTrainer(train_data, val_data, test_data)

    # Smaller configurations for faster training
    transformer_config = {
        'input_dim': 8,
        'd_model': 64,      # Reduced from 256
        'nhead': 4,         # Reduced from 8
        'num_layers': 2,    # Reduced from 4
        'dim_feedforward': 256,  # Reduced from 1024
        'output_dim': 3,
        'dropout': 0.1,
        'seq_length': 48,   # Reduced from 168
        'batch_size': 64,   # Increased from 32
        'learning_rate': 0.001
    }

    patchtst_config = {
        'input_dim': 8,
        'output_dim': 3,
        'patch_len': 12,    # Reduced from 24
        'stride': 6,        # Reduced from 12
        'num_patches': 4,   # Reduced from 7
        'd_model': 64,      # Reduced from 128
        'nhead': 4,         # Reduced from 8
        'num_layers': 2,    # Reduced from 3
        'dim_feedforward': 128,  # Reduced from 256
        'dropout': 0.1,
        'seq_length': 48,   # Reduced from 168
        'batch_size': 64,   # Increased from 32
        'learning_rate': 0.001
    }

    # Reduce number of epochs and patience
    num_epochs = 20    # Reduced from 100
    patience = 5      # Reduced from 10

    # Train and evaluate Transformer
    print("Training Vanilla Transformer...")
    transformer_model, transformer_train_losses, transformer_val_losses = trainer.train_model(
        'transformer', transformer_config, num_epochs=num_epochs, patience=patience)

    # Train and evaluate PatchTST
    print("\nTraining PatchTST...")
    patchtst_model, patchtst_train_losses, patchtst_val_losses = trainer.train_model(
        'patchtst', patchtst_config, num_epochs=num_epochs, patience=patience)

    # Plot and save results
    trainer.plot_training_history('transformer')
    trainer.plot_training_history('patchtst')

    # Print final metrics
    print("\nTransformer Metrics:")
    transformer_metrics, transformer_preds, transformer_actuals = trainer.evaluate_model(
        transformer_model, 'transformer', transformer_config)
    print(json.dumps(transformer_metrics, indent=4))
    print("\nPatchTST Metrics:")
    patchtst_metrics, patchtst_preds, patchtst_actuals = trainer.evaluate_model(
        patchtst_model, 'patchtst', patchtst_config)
    print(json.dumps(patchtst_metrics, indent=4))


if __name__ == "__main__":
    main()
