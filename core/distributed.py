"""Data parallel training with fixed gradient buckets."""

import os
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from core.device import setup_device


class GradientBucket:
    def __init__(self, parameters):
        self.parameters = parameters
        self.buffer = torch.zeros(
            sum(parameter.numel() for parameter in parameters),
            dtype=torch.float32,
            device=parameters[0].device,
        )
        offset = 0
        for parameter in parameters:
            parameter.grad = self.buffer[offset : offset + parameter.numel()].view_as(
                parameter
            )
            offset += parameter.numel()
        self.ready = 0
        self.work = None


class DataParallel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.dp_size = world_size()
        self.sync_gradients = True
        self.next_bucket = 0
        self.buckets = []
        for parameter in module.parameters():
            dist.broadcast(parameter.detach(), src=0)

        parameters = []
        numel = 0
        # Meow's backward reaches the last layers first and tied embedding last.
        for parameter in reversed(list(module.parameters())):
            parameters.append(parameter)
            numel += parameter.numel()
            if numel * 4 >= 25 * 1024 * 1024:
                self.buckets.append(GradientBucket(parameters))
                parameters = []
                numel = 0
        if parameters:
            self.buckets.append(GradientBucket(parameters))
        for bucket in self.buckets:
            for parameter in bucket.parameters:
                parameter.register_post_accumulate_grad_hook(
                    partial(self._gradient_ready, bucket=bucket)
                )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def zero_grad_buffer(self):
        for bucket in self.buckets:
            bucket.buffer.zero_()
            bucket.ready = 0
            bucket.work = None
        self.next_bucket = 0

    @contextmanager
    def no_sync(self):
        self.sync_gradients = False
        try:
            yield
        finally:
            self.sync_gradients = True

    def _gradient_ready(self, parameter, bucket):
        if not self.sync_gradients:
            return
        bucket.ready += 1
        # All ranks issue collectives in the same bucket order.
        while self.next_bucket < len(self.buckets):
            bucket = self.buckets[self.next_bucket]
            if bucket.ready != len(bucket.parameters):
                break
            bucket.buffer.div_(self.dp_size)
            bucket.work = dist.all_reduce(bucket.buffer, async_op=True)
            self.next_bucket += 1

    def finish_grad_sync(self):
        for bucket in self.buckets:
            bucket.work.wait()


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def setup_distributed(name):
    kind = name.split(":", 1)[0]
    if "LOCAL_RANK" in os.environ and kind != "cpu":
        name = f"{kind}:{os.environ['LOCAL_RANK']}"
    device = setup_device(name)
    if "RANK" in os.environ:
        dist.init_process_group(
            backend={"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[kind]
        )
    return device


def wrap_model(model):
    if world_size() == 1:
        return model
    return DataParallel(model)


def unwrap_model(model):
    return model.module if isinstance(model, DataParallel) else model
