import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PatchTST(nn.Module):
    def __init__(self, input_dim, output_dim, patch_len, stride, num_patches,
                 d_model=32, nhead=2, num_layers=1, dim_feedforward=64,
                 dropout=0.1, activation="gelu"):
        super().__init__()

        # Model parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches
        self.d_model = d_model

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            input_dim, d_model, patch_len, stride
        )

        # Positional encoding
        self.pos_embedding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final prediction head
        self.head = nn.Linear(d_model * num_patches, output_dim)

    def forward(self, x):
        # Input shape: [batch_size, seq_length, input_dim]
        batch_size = x.shape[0]

        # Patch embedding: [batch_size, num_patches, d_model]
        x = self.patch_embedding(x)

        # Add positional encoding
        x = self.pos_embedding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Reshape: [batch_size, num_patches * d_model]
        x = x.reshape(batch_size, self.num_patches * self.d_model)

        # Final prediction: [batch_size, output_dim]
        x = self.head(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, patch_len, stride):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        # Linear projection for each patch
        self.projection = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape

        # Create patches using unfold
        # Reshape to [batch_size, input_dim, seq_len] for unfold operation
        x = x.transpose(1, 2)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Reshape patches and project
        patches = patches.reshape(
            batch_size, self.input_dim, -1, self.patch_len)
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(
            batch_size, -1, self.input_dim * self.patch_len)

        # Project patches to d_model dimension
        patches = self.projection(patches)

        return patches


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def create_patchtst_model(config):
    """Create a PatchTST model with the given configuration."""
    model = PatchTST(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        patch_len=config['patch_len'],
        stride=config['stride'],
        num_patches=config['num_patches'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config.get('dropout', 0.1)
    )
    return model


# Example configuration
model_config = {
    'input_dim': 8,          # Number of features
    'output_dim': 3,         # Number of target variables
    'patch_len': 16,         # Length of each patch
    'stride': 8,             # Stride between patches
    'num_patches': 8,        # Number of patches
    'd_model': 128,          # Transformer model dimension
    'nhead': 8,              # Number of attention heads
    'num_layers': 3,         # Number of transformer layers
    'dim_feedforward': 256,  # Dimension of feedforward network
    'dropout': 0.1           # Dropout rate
}


def prepare_data_patchtst(data, seq_length, pred_length=1):
    """
    Prepare sequences for PatchTST model.
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


def train_patchtst(model, train_loader, criterion, optimizer, device):
    """Training function for PatchTST model."""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_patchtst(model, val_loader, criterion, device):
    """Evaluation function for PatchTST model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)
