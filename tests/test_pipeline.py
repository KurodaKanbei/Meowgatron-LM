from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from core.config import ModelConfig
from core.distributed import (
    data_parallel_rank,
    pipeline_rank,
    setup_parallel,
    wrap_model,
)
from core.model import Meow
from core.pipeline import PipelineStage, pipeline_step, sync_shared_gradients

MODEL = ModelConfig(
    vocab_size=32,
    hidden_size=32,
    num_heads=4,
    mlp_hidden_size=128,
    max_seq_len=8,
    num_layers=4,
)
MICROBATCHES = 4


def input_batches(dp):
    inputs = torch.arange(MICROBATCHES * 2 * dp * 8).reshape(-1, 8)
    return (inputs + torch.arange(inputs.size(0))[:, None]) % MODEL.vocab_size


def pipeline_worker(rank, layout, dp, recompute, rendezvous, output):
    torch.set_num_threads(1)
    pp = len(layout)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=pp * dp,
        timeout=timedelta(seconds=60),
    )
    try:
        setup_parallel(pp)
        torch.manual_seed(31)
        stage = PipelineStage(
            Meow(MODEL, recompute, recompute_modules=("attention", "norm", "mlp_act")),
            pipeline_rank(),
            layout,
        )
        model = wrap_model(stage)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        events = []
        stage.register_forward_hook(lambda *_: events.append("F"))
        next(stage.parameters()).register_post_accumulate_grad_hook(
            lambda _: events.append("B")
        )
        for step in range(2):
            events.clear()
            inputs = input_batches(dp).roll(step, dims=1)[data_parallel_rank() :: dp]
            microbatches = [
                (batch, (batch + 3) % MODEL.vocab_size) for batch in inputs.split(2)
            ]
            if dp > 1:
                model.zero_grad_buffer()
            else:
                optimizer.zero_grad(set_to_none=True)
            loss = pipeline_step(model, microbatches, torch.device("cpu"), "fp32")
            if dp > 1:
                model.finish_grad_sync()
            sync_shared_gradients(model)
            optimizer.step()
        torch.save(
            {
                "model": stage.state_dict(),
                "gradients": {
                    name: parameter.grad for name, parameter in stage.named_parameters()
                },
                "loss": loss / MICROBATCHES,
                "events": events,
            },
            Path(output) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "layout,dp,recompute",
    [([1, 3], 1, "full"), ([1, 1, 1, 1], 1, "selective"), ([1, 3], 2, "none")],
)
def test_pipeline_matches_unpartitioned_updates(
    tmp_path, monkeypatch, layout, dp, recompute
):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    pp = len(layout)
    mp.spawn(
        pipeline_worker,
        args=(layout, dp, recompute, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=pp * dp,
    )
    torch.manual_seed(31)
    model = Meow(MODEL)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        inputs = input_batches(dp).roll(step, dims=1)
        for data_rank in range(dp):
            local_loss = 0.0
            for batch in inputs[data_rank::dp].split(2):
                loss = F.cross_entropy(
                    model(batch).flatten(0, 1),
                    ((batch + 3) % MODEL.vocab_size).flatten(),
                )
                (loss / (MICROBATCHES * dp)).backward()
                local_loss += loss.detach() / MICROBATCHES
            losses.append(local_loss)
        optimizer.step()
    reference_weights = model.state_dict()
    reference_gradients = {
        name: parameter.grad for name, parameter in model.named_parameters()
    }
    for rank in range(pp * dp):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        for name, weight in actual["model"].items():
            torch.testing.assert_close(
                weight, reference_weights[name], rtol=1e-5, atol=1e-6
            )
        for name, gradient in actual["gradients"].items():
            if name == "lm_head.weight":
                name = "token_embedding.weight"
            torch.testing.assert_close(
                gradient, reference_gradients[name], rtol=1e-5, atol=1e-6
            )
        stage_index = rank % pp
        if stage_index == pp - 1:
            torch.testing.assert_close(actual["loss"], losses[rank // pp])
        outstanding = 0
        peak = 0
        for event in actual["events"]:
            outstanding += 1 if event == "F" else -1
            peak = max(peak, outstanding)
        assert outstanding == 0
        assert peak == min(pp - stage_index, MICROBATCHES)
