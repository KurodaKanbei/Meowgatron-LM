"""Assemble and train the first Meow model on a local FineWeb-Edu snapshot."""

import argparse
import importlib.metadata
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from core.checkpoint import load_checkpoint
from core.data import TokenDataset
from core.distributed import rank, setup_distributed, world_size, wrap_model
from core.model import Meow
from core.resolver import config_to_yaml, load_config
from core.training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meow.yaml"),
        help="experiment YAML (default: configs/meow.yaml)",
    )
    parser.add_argument(
        "overrides", nargs="*", metavar="key=value", help="optional YAML overrides"
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, args.overrides)
    device = setup_distributed(config.device)
    try:
        run_experiment(config_path, config, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_experiment(config_path, config, device):
    local_batch = config.data.micro_batch_size * world_size()
    if (
        local_batch <= 0
        or config.data.global_batch_size <= 0
        or config.data.global_batch_size % local_batch
    ):
        raise ValueError(
            "GBS must be a positive multiple of per-rank MBS * DP world size"
        )
    data_root = Path(config.data.root)
    tokenizer_path = data_root / "tokenizers/gpt2/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    dataset = TokenDataset(
        (data_root / "fineweb-edu-10bt/sample/10BT").glob("*.parquet"),
        tokenizer,
        config.model.max_seq_len,
        rank=rank(),
        world_size=world_size(),
    )
    if not dataset.files:
        raise FileNotFoundError(f"No FineWeb-Edu Parquet shards under {data_root}")
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.micro_batch_size,
        num_workers=0,
        drop_last=True,
    )
    torch.manual_seed(config.seed)
    model = Meow(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), **asdict(config.optimizer), foreach=False, fused=False
    )

    output = [None]
    if rank() == 0:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        output[0] = str(Path(config.output_root) / f"{config_path.stem}-{timestamp}")
        Path(output[0]).mkdir(parents=True, exist_ok=False)
    if world_size() > 1:
        dist.broadcast_object_list(output, src=0)
    output_dir = Path(output[0])
    resolved_yaml = config_to_yaml(config)
    run = {
        "config": asdict(config),
        "config_source": str(config_path),
        "device": str(device),
        "world_size": world_size(),
        "parameter_dtype": "float32",
        "torch_version": torch.__version__,
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "data_files": [str(path.relative_to(data_root)) for path in dataset.files],
        "data_order": "sorted shards, stored rows, packed sample i goes to rank i % DP",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    manifest_path = data_root / "fineweb-edu-10bt/manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        run["dataset_revision"] = manifest.get("revision")
    if device.type == "npu":
        run["torch_npu_version"] = importlib.metadata.version("torch_npu")
    progress = None
    if config.checkpoint.resume is not None:
        progress = load_checkpoint(config.checkpoint.resume, model, optimizer, dataset)
    model = wrap_model(model)
    if rank() == 0:
        (output_dir / "resolved.yaml").write_text(resolved_yaml, encoding="utf-8")
        (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        print(
            f"Meow | {run['parameters']:,} parameters | {device}"
            f" | {config.training.precision} | dp={world_size()}"
            f" | mbs={config.data.micro_batch_size} | gbs={config.data.global_batch_size}"
        )
    writer_context = nullcontext(None)
    if config.tensorboard.enabled and rank() == 0:
        from torch.utils.tensorboard import SummaryWriter

        writer_context = SummaryWriter(
            log_dir=str(output_dir / "tensorboard"),
            flush_secs=config.tensorboard.flush_secs,
        )
    with writer_context as writer:
        if writer is not None:
            writer.add_text("config/resolved", f"```yaml\n{resolved_yaml}```", 0)
        result = train(
            model,
            dataloader,
            optimizer,
            config.training,
            device,
            writer,
            checkpoint_path=output_dir / "checkpoint.pt",
            checkpoint_interval=config.checkpoint.interval,
            gradient_accumulation_steps=config.data.global_batch_size // local_batch,
            progress=progress,
        )
    run["result"] = result
    if rank() == 0:
        (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        print(f"Finished at step {result['steps']}; artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
