"""Save an optimizer-update boundary and restore its next input on every rank."""

import shutil
from pathlib import Path

import torch
import torch.distributed as dist

from core.distributed import (
    data_parallel_group,
    data_parallel_rank,
    data_parallel_size,
    pipeline_rank,
    pipeline_size,
    rank,
    unwrap_model,
    world_size,
)


def _checkpoint_file(path, step=None):
    path = Path(path)
    if pipeline_size() == 1:
        return path
    if step is None:
        step = int((path / "latest").read_text())
    return path / f"step-{step}" / f"stage-{pipeline_rank()}.pt"


def save_checkpoint(path, model, optimizer, dataset, progress):
    """DP shares one copy; each pipeline stage writes its own state."""
    root = Path(path)
    data_state = dataset.state_dict()
    data_states = [data_state]
    if data_parallel_size() > 1:
        data_states = [None] * data_parallel_size()
        dist.all_gather_object(data_states, data_state, group=data_parallel_group())
    if data_parallel_rank() == 0:
        path = _checkpoint_file(root, progress["steps"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        torch.save(
            {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "progress": progress,
                "data": data_states,
            },
            temporary_path,
        )
        temporary_path.replace(path)
    if world_size() > 1:
        dist.barrier()
    if pipeline_size() > 1:
        if rank() == 0:
            latest = root / "latest"
            previous = int(latest.read_text()) if latest.exists() else None
            temporary = root / "latest.tmp"
            temporary.write_text(str(progress["steps"]))
            temporary.replace(latest)
            if previous is not None and previous != progress["steps"]:
                shutil.rmtree(root / f"step-{previous}")
        dist.barrier()


def load_checkpoint(path, model, optimizer, dataset):
    """Restore existing Parameters, including the shared embedding/LM head."""
    checkpoint = torch.load(
        _checkpoint_file(path), map_location="cpu", weights_only=True
    )
    if len(checkpoint["data"]) != data_parallel_size():
        raise ValueError("Resume requires the same DP world size")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    # Native load_state_dict moves AdamW moments to each Parameter's device.
    optimizer.load_state_dict(checkpoint["optimizer"])
    dataset.load_state_dict(checkpoint["data"][data_parallel_rank()])
    return checkpoint["progress"]
