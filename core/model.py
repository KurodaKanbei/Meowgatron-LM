"""Meow: pre-norm Attention + dense MLP blocks for next-token prediction."""

import torch
from torch import nn

from core.attention import SelfAttention
from core.config import ModelConfig
from core.mlp import DenseMLP


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.attention = SelfAttention(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = DenseMLP(config)

    def forward(self, hidden):
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class Meow(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    @torch.no_grad()
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.size(1)
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.final_norm(hidden))
