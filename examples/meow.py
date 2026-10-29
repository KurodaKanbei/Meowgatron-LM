"""Assemble and train the first Meow model on a local FineWeb-Edu snapshot."""

import argparse
import hashlib
import importlib.metadata
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from core.data import TokenDataset
from core.device import setup_device
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

    data_root = Path(config.data.root)
    tokenizer_path = data_root / "tokenizers/gpt2/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    dataset = TokenDataset(
        (data_root / "fineweb-edu-10bt/sample/10BT").glob("*.parquet"),
        tokenizer,
        config.model.max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.micro_batch_size,
        num_workers=0,
        drop_last=True,
    )
    device = setup_device(config.device)
    torch.manual_seed(config.seed)
    model = Meow(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), **asdict(config.optimizer), foreach=False, fused=False
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output_dir = Path(config.output_root) / f"{config_path.stem}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_yaml = config_to_yaml(config)
    (output_dir / "resolved.yaml").write_text(resolved_yaml, encoding="utf-8")
    run = {
        "config": asdict(config),
        "config_source": str(config_path),
        "config_sha256": hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest(),
        "device": str(device),
        "optimizer": {
            "name": "AdamW",
            **asdict(config.optimizer),
            "foreach": False,
            "fused": False,
        },
        "parameter_dtype": "float32",
        "torch_version": torch.__version__,
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "data_root": str(data_root),
        "data_files": [str(path.relative_to(data_root)) for path in dataset.files],
        "data_order": "sorted shards, stored row order, no shuffle",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    manifest_path = data_root / "fineweb-edu-10bt/manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        run["dataset_revision"] = manifest.get("revision")
        run["dataset_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    if device.type == "npu":
        run["torch_npu_version"] = importlib.metadata.version("torch_npu")
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(
        f"Meow | {run['parameters']:,} parameters | {device} | {config.training.precision}"
        f" | mbs={config.data.micro_batch_size} | gbs={config.data.global_batch_size}"
    )
    writer_context = nullcontext(None)
    if config.tensorboard.enabled:
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
            gradient_accumulation_steps=(
                config.data.global_batch_size // config.data.micro_batch_size
            ),
        )
    run["result"] = result
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    # This is a portable model snapshot, not an exact training-resume checkpoint.
    torch.save(
        {
            "model": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "config": asdict(config.model),
            "steps": result["steps"],
        },
        output_dir / "model.pt",
    )
    print(f"Finished {result['steps']} steps; artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
