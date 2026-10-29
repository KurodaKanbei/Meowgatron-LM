"""Causal multi-head self-attention, written out with native PyTorch ops."""

import math

import torch
from torch import nn

from core.config import ModelConfig


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = (query @ key.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        probabilities = scores.softmax(dim=-1).to(value.dtype)
        context = (probabilities @ value).transpose(1, 2).reshape(batch, length, width)
        return self.proj(context)
