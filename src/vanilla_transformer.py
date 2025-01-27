import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (won't be updated during training)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim,
                 dropout=0.1, max_seq_length=5000):
        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, src, src_mask=None):
        # Project input to d_model dimensions
        x = self.input_projection(src)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_mask)

        # Project to output dimensions
        output = self.output_projection(x)

        return output


def create_transformer_model(config):
    """Create a transformer model with the given configuration."""
    model = TimeSeriesTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        output_dim=config['output_dim'],
        dropout=config.get('dropout', 0.1)
    )
    return model


# Example configuration
model_config = {
    'input_dim': 8,  # Number of features in input
    'd_model': 256,  # Transformer model dimension
    'nhead': 8,      # Number of attention heads
    'num_layers': 4,  # Number of transformer layers
    'dim_feedforward': 1024,  # Dimension of feedforward network
    'output_dim': 3,  # Number of target variables (Zone1, Zone2, Zone3)
    'dropout': 0.1   # Dropout rate
}

# Create sequence mask for training


def create_mask(seq_length):
    """Create mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    return mask


def prepare_data(data, seq_length, pred_length=1):
    """
    Prepare sequences for transformer model.
    Args:
        data: Input data array
        seq_length: Length of input sequence
        pred_length: Length of prediction sequence (default=1)
    Returns:
        - input sequences
        - target sequences
    """
    sequences = []
    targets = []

    # Convert data to numpy if it's a tensor
    if torch.is_tensor(data):
        data = data.cpu().numpy()

    for i in range(len(data) - seq_length - pred_length + 1):
        # Get sequence of features
        seq = data[i:i+seq_length]
        # Get target sequence (last 3 columns)
        if pred_length == 1:
            target = data[i+seq_length, -3:]  # Single step prediction
        else:
            target = data[i+seq_length:i+seq_length +
                          pred_length, -3:]  # Multi-step prediction

        sequences.append(seq)
        targets.append(target)

    # Convert lists to numpy arrays first
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    # Convert to tensors
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

# Training function


def train_transformer(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Create mask for transformer
        mask = create_mask(data.size(1)).to(device)

        optimizer.zero_grad()
        output = model(data, mask)

        # Get predictions for the last time step
        predictions = output[:, -1, :]
        loss = criterion(predictions, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Evaluation function


def evaluate_transformer(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            mask = create_mask(data.size(1)).to(device)

            output = model(data, mask)
            predictions = output[:, -1, :]
            loss = criterion(predictions, target)

            total_loss += loss.item()

    return total_loss / len(val_loader)
