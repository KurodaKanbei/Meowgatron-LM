"""Save an optimizer-update boundary and restore its next input on every rank."""

from pathlib import Path

import torch
import torch.distributed as dist

from core.distributed import rank, unwrap_model, world_size


def save_checkpoint(path, model, optimizer, dataset, progress):
    """All ranks participate; rank 0 atomically replaces one shared file."""
    data_state = dataset.state_dict()
    data_states = [data_state]
    if world_size() > 1:
        data_states = [None] * world_size()
        dist.all_gather_object(data_states, data_state)
    if rank() == 0:
        path = Path(path)
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


def load_checkpoint(path, model, optimizer, dataset):
    """Restore existing Parameters, including the shared embedding/LM head."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if len(checkpoint["data"]) != world_size():
        raise ValueError("Resume requires the same DP world size")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    # Native load_state_dict moves AdamW moments to each Parameter's device.
    optimizer.load_state_dict(checkpoint["optimizer"])
    dataset.load_state_dict(checkpoint["data"][rank()])
    return checkpoint["progress"]
