"""Causal multi-head self-attention, written out with native PyTorch ops."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from core.config import ModelConfig


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, recompute: bool = False):
        super().__init__()
        self.recompute = recompute
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
        return self._forward_qkv(self.qkv(hidden))

    def _forward_qkv(self, qkv):
        batch, length, _ = qkv.shape
        width = self.num_heads * self.head_dim
        qkv = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if self.recompute:
            context = checkpoint(
                self._attention,
                query,
                key,
                value,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            context = self._attention(query, key, value)
        return self.proj(context.transpose(1, 2).reshape(batch, length, width))

    def _attention(self, query, key, value):
        length = query.size(-2)
        scores = (query @ key.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        probabilities = scores.softmax(dim=-1).to(value.dtype)
        return probabilities @ value
