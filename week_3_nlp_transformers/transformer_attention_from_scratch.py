"""Scaled dot-product self-attention implemented with PyTorch primitives."""

import math
import torch
from torch import Tensor, nn


class ScaledDotProductSelfAttention(nn.Module):
    """Multi-head self-attention with an optional boolean padding mask."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads != 0 or not 0 <= dropout < 1:
            raise ValueError("embed_dim must be divisible by num_heads; dropout must be in [0, 1)")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.query, self.key, self.value = (nn.Linear(embed_dim, embed_dim) for _ in range(3))
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.embed_dim:
            raise ValueError("hidden_states must have shape (batch, sequence, embed_dim)")
        batch, sequence, _ = hidden_states.shape
        def split_heads(values: Tensor) -> Tensor:
            return values.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        queries, keys, values = (split_heads(layer(hidden_states)) for layer in (self.query, self.key, self.value))
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            if padding_mask.shape != (batch, sequence):
                raise ValueError("padding_mask must match (batch, sequence)")
            scores = scores.masked_fill(~padding_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = (weights @ values).transpose(1, 2).contiguous().view(batch, sequence, self.embed_dim)
        return self.output(context), weights
