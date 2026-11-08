"""Data parallel training with fixed gradient buckets."""

import os
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from core.device import setup_device

_data_parallel_group = None
_pipeline_group = None
_embedding_group = None
_pipeline_ranks = []
_pipeline_size = 1


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
        self.dp_size = data_parallel_size()
        self.sync_gradients = True
        self.next_bucket = 0
        self.buckets = []
        for parameter in module.parameters():
            dist.broadcast(
                parameter.detach(), src=pipeline_rank(), group=data_parallel_group()
            )

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
            bucket.work = dist.all_reduce(
                bucket.buffer, group=data_parallel_group(), async_op=True
            )
            self.next_bucket += 1

    def finish_grad_sync(self):
        for bucket in self.buckets:
            bucket.work.wait()


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def data_parallel_group():
    return _data_parallel_group


def data_parallel_size():
    return world_size() // _pipeline_size


def data_parallel_rank():
    return rank() // _pipeline_size


def pipeline_group():
    return _pipeline_group


def pipeline_size():
    return _pipeline_size


def pipeline_rank():
    return rank() % _pipeline_size


def pipeline_ranks():
    return _pipeline_ranks


def embedding_group():
    return _embedding_group


def setup_parallel(size):
    global _data_parallel_group, _pipeline_group, _embedding_group
    global _pipeline_ranks, _pipeline_size
    if size < 1 or world_size() % size:
        raise ValueError("World size must be a multiple of pipeline_size")
    _pipeline_size = size
    _data_parallel_group = _pipeline_group = _embedding_group = None
    _pipeline_ranks = [rank()]
    if size == 1:
        return
    for stage in range(size):
        ranks = list(range(stage, world_size(), size))
        group = dist.new_group(ranks)
        if rank() in ranks:
            _data_parallel_group = group
    for start in range(0, world_size(), size):
        ranks = list(range(start, start + size))
        group = dist.new_group(ranks)
        shared_group = dist.new_group([ranks[0], ranks[-1]])
        if rank() in ranks:
            _pipeline_group = group
            _pipeline_ranks = ranks
        if rank() in (ranks[0], ranks[-1]):
            _embedding_group = shared_group
    # Initialize the communicator before pairwise P2P operations use it.
    dist.barrier(group=_pipeline_group)


def setup_distributed(name, pipeline_size=1):
    kind = name.split(":", 1)[0]
    if "LOCAL_RANK" in os.environ and kind != "cpu":
        name = f"{kind}:{os.environ['LOCAL_RANK']}"
    device = setup_device(name)
    if "RANK" in os.environ:
        dist.init_process_group(
            backend={"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[kind]
        )
    setup_parallel(pipeline_size)
    return device


def wrap_model(model):
    if data_parallel_size() == 1:
        return model
    return DataParallel(model)


def unwrap_model(model):
    return model.module if isinstance(model, DataParallel) else model
