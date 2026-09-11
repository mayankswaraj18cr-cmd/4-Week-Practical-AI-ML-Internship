"""Configurable feed-forward network and deterministic data loaders."""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class FeedForwardNetwork(nn.Module):
    """MLP with normalization and dropout for tabular features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) < 1 or not 0 <= dropout < 1:
            raise ValueError("dimensions must be positive and dropout must be in [0, 1)")
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape (batch, features)")
        return self.network(features)


def make_dataloader(features: Tensor, targets: Tensor, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create a validated tensor DataLoader."""
    if features.ndim != 2 or targets.shape[0] != features.shape[0] or batch_size < 1:
        raise ValueError("features, targets, and batch_size are incompatible")
    return DataLoader(TensorDataset(features.float(), targets), batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    """Train one epoch and return mean loss."""
    model.train()
    total, count = 0.0, 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * features.size(0)
        count += features.size(0)
    return total / count if count else 0.0
