"""Contiguous pipeline stages and a non-interleaved 1F1B schedule."""

from collections import deque
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from core.distributed import (
    DataParallel,
    embedding_group,
    pipeline_group,
    pipeline_rank,
    pipeline_ranks,
    pipeline_size,
    unwrap_model,
)


class PipelineStage(nn.Module):
    def __init__(self, model, stage_index, layout):
        super().__init__()
        self.is_first = stage_index == 0
        self.is_last = stage_index == len(layout) - 1
        self.hidden_size = model.token_embedding.embedding_dim
        first_layer = sum(layout[:stage_index])
        if self.is_first:
            self.token_embedding = model.token_embedding
            self.position_embedding = model.position_embedding
        # Keep global layer numbers in checkpoint keys.
        self.layers = nn.ModuleDict(
            {
                str(index): model.layers[index]
                for index in range(first_layer, first_layer + layout[stage_index])
            }
        )
        if self.is_last:
            self.final_norm = model.final_norm
            self.lm_head = model.lm_head

    @property
    def shared_weight(self):
        if self.is_first:
            return self.token_embedding.weight
        if self.is_last:
            return self.lm_head.weight
        return None

    def forward(self, hidden):
        if self.is_first:
            positions = torch.arange(hidden.size(1), device=hidden.device)
            hidden = self.token_embedding(hidden) + self.position_embedding(positions)
        for layer in self.layers.values():
            hidden = layer(hidden)
        if self.is_last:
            hidden = self.lm_head(self.final_norm(hidden))
        return hidden


def sync_shared_gradients(model):
    """Embedding and LM head contribute to the same optimizer update."""
    weight = unwrap_model(model).shared_weight
    if weight is not None:
        dist.all_reduce(weight.grad, group=embedding_group())


def pipeline_step(model, microbatches, device, precision):
    """Run forward/backward for one update, leaving gradient sync to the caller."""
    stage = unwrap_model(model)
    index = pipeline_rank()
    ranks = pipeline_ranks()
    count = len(microbatches)
    shape = (*microbatches[0][0].shape, stage.hidden_size)
    pending = deque()
    step_loss = torch.zeros((), device=device)

    def communicate(send=None, receive=False, forward=True):
        # Pair the opposing directions before waiting; send-then-recv can deadlock.
        operations = []
        if send is not None:
            peer = ranks[index + (1 if forward else -1)]
            operations.append(
                dist.P2POp(dist.isend, send.detach(), peer, group=pipeline_group())
            )
        incoming = None
        if receive:
            peer = ranks[index + (1 if forward else -1)]
            incoming = torch.empty(shape, dtype=torch.float32, device=device)
            operations.append(
                dist.P2POp(dist.irecv, incoming, peer, group=pipeline_group())
            )
        if operations:
            for work in dist.batch_isend_irecv(operations):
                work.wait()
        return incoming

    def forward(microbatch, hidden):
        nonlocal step_loss
        inputs, targets = microbatches[microbatch]
        if stage.is_first:
            hidden = inputs.to(device)
        else:
            hidden.requires_grad_()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            output = model(hidden)
        if stage.is_last:
            targets = targets.to(device)
            loss = F.cross_entropy(output.float().flatten(0, 1), targets.flatten())
            step_loss += loss.detach()
            output = loss / count
        pending.append((hidden, output))
        return output

    def backward(microbatch, gradient):
        hidden, output = pending.popleft()
        context = (
            model.no_sync()
            if isinstance(model, DataParallel) and microbatch < count - 1
            else nullcontext()
        )
        with context:
            torch.autograd.backward(output, gradient)
        return None if stage.is_first else hidden.grad

    warmup = min(pipeline_size() - index - 1, count)
    for microbatch in range(warmup):
        hidden = communicate(receive=not stage.is_first, forward=False)
        output = forward(microbatch, hidden)
        communicate(send=output)

    remaining = count - warmup
    hidden = (
        communicate(receive=not stage.is_first, forward=False) if remaining else None
    )
    for microbatch in range(remaining):
        output = forward(microbatch + warmup, hidden)
        gradient = communicate(
            send=None if stage.is_last else output,
            receive=not stage.is_last,
        )
        input_gradient = backward(microbatch, gradient)
        hidden = communicate(
            send=input_gradient,
            receive=not stage.is_first and microbatch < remaining - 1,
            forward=False,
        )

    for microbatch in range(remaining, count):
        gradient = communicate(receive=not stage.is_last)
        input_gradient = backward(microbatch, gradient)
        communicate(send=input_gradient, forward=False)
    return step_loss
