"""Resolve YAML into the small set of configuration objects used by Meow."""

from dataclasses import asdict, replace
from pathlib import Path

from omegaconf import OmegaConf

from core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TensorBoardConfig,
    TrainConfig,
)

OmegaConf.register_resolver("meow.mul", lambda left, right: left * right)


def load_config(path: str | Path, overrides=()) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    merged = OmegaConf.merge(
        OmegaConf.create(asdict(ExperimentConfig())),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(list(overrides)),
    )
    values = OmegaConf.to_container(merged, resolve=True)
    for section, factory in (
        ("data", DataConfig),
        ("model", ModelConfig),
        ("optimizer", OptimizerConfig),
        ("training", TrainConfig),
        ("tensorboard", TensorBoardConfig),
    ):
        values[section] = factory(**values[section])
    config = ExperimentConfig(**values)

    def absolute_path(value):
        return str((path.parent / Path(value).expanduser()).resolve())

    return replace(
        config,
        data=replace(config.data, root=absolute_path(config.data.root)),
        output_root=absolute_path(config.output_root),
    )


def config_to_yaml(config: ExperimentConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.create(asdict(config)))
