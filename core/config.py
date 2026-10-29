"""The configuration of our first model and training run."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50_257
    hidden_size: int = 128
    num_heads: int = 4
    mlp_hidden_size: int = 512
    max_seq_len: int = 128
    num_layers: int = 1


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int | None = 100
    precision: str = "fp32"
    log_interval: int = 10
    checkpoint_interval: int | None = None


@dataclass(frozen=True)
class DataConfig:
    root: str = "../data"
    micro_batch_size: int = 4
    global_batch_size: int = 4


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 0.0003
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TensorBoardConfig:
    enabled: bool = True
    flush_secs: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    device: str = "npu:0"
    seed: int = 42
    output_root: str = "../runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
