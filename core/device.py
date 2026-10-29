"""Device setup and synchronization; the model only uses torch operations."""

import importlib

import torch


def setup_device(name: str) -> torch.device:
    kind = name.split(":", 1)[0]
    if kind == "npu":
        importlib.import_module("torch_npu")
    device = torch.device(name)
    if kind != "cpu":
        backend = getattr(torch, kind)
        backend.set_device(device)
    if kind == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    if kind == "npu":
        torch.npu.matmul.allow_hf32 = False
    return device


def synchronize(device: torch.device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)
