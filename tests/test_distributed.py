from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from core.config import ModelConfig, TrainConfig
from core.distributed import wrap_model
from core.model import Meow
from core.training import train

MODEL = ModelConfig(
    vocab_size=8,
    hidden_size=256,
    num_heads=4,
    mlp_hidden_size=1024,
    max_seq_len=8,
    num_layers=12,
)


def gradient_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(31 + rank)
        model = Meow(MODEL)
        ready = set()
        communication = []
        forward = 0

        def gradient_ready(parameter):
            ready.add(parameter)

        for parameter in model.parameters():
            parameter.register_post_accumulate_grad_hook(gradient_ready)
        dp = wrap_model(model)

        all_reduce = dist.all_reduce

        def communicate(tensor, *args, **kwargs):
            if kwargs.get("async_op", False):
                communication.append((forward, len(ready)))
            return all_reduce(tensor, *args, **kwargs)

        def begin_backward(*_):
            nonlocal forward
            forward += 1
            ready.clear()

        dp.register_forward_pre_hook(begin_backward)
        optimizer = torch.optim.SGD(dp.parameters(), lr=0.01)
        inputs = (torch.arange(64).reshape(8, 8) + torch.arange(8)[:, None]) % 8
        inputs = inputs[rank::2]
        targets = (inputs + 3) % 8
        with patch.object(dist, "all_reduce", communicate):
            result = train(
                dp,
                [(inputs[:2], targets[:2]), (inputs[2:], targets[2:])] * 2,
                optimizer,
                TrainConfig(max_steps=2),
                torch.device("cpu"),
                gradient_accumulation_steps=2,
            )
        torch.save(
            {
                "model": model.state_dict(),
                "grads": [p.grad for p in model.parameters()],
                "loss": result["final_loss"],
                "tokens_seen": result["tokens_seen"],
                "communication": communication,
            },
            Path(output) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_dp_gradient_average_matches_single_rank_training(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    mp.spawn(
        gradient_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2
    )
    torch.manual_seed(31)
    model = Meow(MODEL)
    inputs = (torch.arange(64).reshape(8, 8) + torch.arange(8)[:, None]) % 8
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(inputs).flatten(0, 1), ((inputs + 3) % 8).flatten()
        )
        loss.backward()
        optimizer.step()
    reference = {
        "model": model.state_dict(),
        "grads": [p.grad for p in model.parameters()],
        "loss": loss.item(),
        "tokens_seen": 128,
    }
    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for result in (rank0, rank1):
        communication = result.pop("communication")
        assert {forward for forward, _ in communication} == {2, 4}
        for forward in (2, 4):
            launched = [ready for step, ready in communication if step == forward]
            assert len(launched) > 1
            assert launched[0] < len(list(model.parameters()))
    torch.testing.assert_close(rank0, rank1, rtol=0, atol=0)
    torch.testing.assert_close(rank0, reference, rtol=1e-5, atol=1e-6)
