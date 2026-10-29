"""A dense feed-forward layer."""

from torch import nn

from core.config import ModelConfig


class DenseMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.mlp_hidden_size, bias=False)
        self.activation = nn.GELU()
        self.down = nn.Linear(config.mlp_hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down(self.activation(self.up(hidden)))
