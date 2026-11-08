"""A dense feed-forward layer."""

from torch import nn
from torch.utils.checkpoint import checkpoint

from core.config import ModelConfig


class DenseMLP(nn.Module):
    def __init__(self, config: ModelConfig, recompute: bool = False):
        super().__init__()
        self.recompute = recompute
        self.up = nn.Linear(config.hidden_size, config.mlp_hidden_size, bias=False)
        self.activation = nn.GELU()
        self.down = nn.Linear(config.mlp_hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self._forward_up(self.up(hidden))

    def _forward_up(self, hidden):
        if self.recompute:
            # Include the consumer so its saved GELU output can be discarded.
            # Non-reentrant early-stop skips the down GEMM during recomputation.
            return checkpoint(
                self._activate_and_project,
                hidden,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._activate_and_project(hidden)

    def _activate_and_project(self, hidden):
        return self.down(self.activation(hidden))
