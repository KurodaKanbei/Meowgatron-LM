"""Meow: pre-norm Attention + dense MLP blocks for next-token prediction."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from core.attention import SelfAttention
from core.config import ModelConfig
from core.mlp import DenseMLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        recompute: str = "none",
        recompute_modules=("attention", "mlp_act"),
    ):
        super().__init__()
        self.recompute = recompute
        selected = recompute_modules if recompute == "selective" else ()
        self.recompute_norm = "norm" in selected
        self.recompute_mlp = "mlp" in selected
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.attention = SelfAttention(config, recompute="attention" in selected)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = DenseMLP(
            config, recompute="mlp_act" in selected and not self.recompute_mlp
        )

    def forward(self, hidden):
        if self.recompute == "full":
            return checkpoint(
                self._forward, hidden, use_reentrant=False, preserve_rng_state=False
            )
        return self._forward(hidden)

    def _forward(self, hidden):
        if self.recompute_norm:
            qkv = checkpoint(
                self._norm_qkv, hidden, use_reentrant=False, preserve_rng_state=False
            )
            attention_output = self.attention._forward_qkv(qkv)
        else:
            attention_output = self.attention(self.attention_norm(hidden))
        hidden = hidden + attention_output

        if self.recompute_mlp:
            # Recompute norm with the MLP when both regions are selected.
            mlp_output = checkpoint(
                self._norm_mlp if self.recompute_norm else self.mlp,
                hidden if self.recompute_norm else self.mlp_norm(hidden),
                use_reentrant=False,
                preserve_rng_state=False,
            )
        elif self.recompute_norm:
            projected = checkpoint(
                self._norm_up, hidden, use_reentrant=False, preserve_rng_state=False
            )
            mlp_output = self.mlp._forward_up(projected)
        else:
            mlp_output = self.mlp(self.mlp_norm(hidden))
        return hidden + mlp_output

    def _norm_qkv(self, hidden):
        # Include the consumer; native early-stop skips the projection GEMM.
        return self.attention.qkv(self.attention_norm(hidden))

    def _norm_up(self, hidden):
        return self.mlp.up(self.mlp_norm(hidden))

    def _norm_mlp(self, hidden):
        return self.mlp(self.mlp_norm(hidden))


class Meow(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        recompute: str = "none",
        recompute_modules=("attention", "mlp_act"),
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, recompute, recompute_modules)
                for _ in range(config.num_layers)
            ]
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
