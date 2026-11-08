"""Native training with gradient accumulation, DP and update-boundary checkpoints."""

import time
from contextlib import nullcontext
from itertools import islice

import torch
import torch.distributed as dist
from torch.nn import functional as F

from core.checkpoint import save_checkpoint
from core.config import TrainConfig
from core.device import synchronize
from core.distributed import rank, world_size


def train(
    model,
    dataloader,
    optimizer,
    config: TrainConfig,
    device: torch.device,
    writer=None,
    checkpoint_path=None,
    gradient_accumulation_steps=1,
    checkpoint_interval=None,
    progress=None,
):
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError("checkpoint.interval must be positive or null")
    progress = (
        dict(progress) if progress is not None else {"steps": 0, "tokens_seen": 0}
    )
    model.train()
    synchronize(device)
    started = time.perf_counter()
    initial_steps = progress["steps"]
    initial_tokens = progress["tokens_seen"]
    first_loss = None
    final_loss = None
    iteration_seconds = 0.0
    interval_seconds = 0.0
    interval_steps = 0
    interval_tokens = 0
    last_saved_step = None
    dataset = dataloader.dataset if checkpoint_path is not None else None
    batches = iter(dataloader)

    while config.max_steps is None or progress["steps"] < config.max_steps:
        # No workers/prefetch: this cursor is exactly the completed-update boundary.
        data_before_step = dataset.state_dict() if dataset is not None else None
        iteration_started = time.perf_counter()
        microbatches = list(islice(batches, gradient_accumulation_steps))
        ready = len(microbatches) == gradient_accumulation_steps
        if world_size() > 1:
            all_ready = torch.tensor(int(ready), dtype=torch.int32, device=device)
            dist.all_reduce(all_ready, op=dist.ReduceOp.MIN)
            ready = bool(all_ready.item())
        if not ready:
            # Discard an incomplete global batch everywhere, keeping the final
            # checkpoint at the last update that actually changed parameters.
            if dataset is not None:
                dataset.load_state_dict(data_before_step)
            break
        if world_size() > 1:
            model.zero_grad_buffer()
        else:
            optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        for micro_step, (inputs, targets) in enumerate(microbatches):
            inputs = inputs.to(device)
            targets = targets.to(device)
            sync_context = (
                model.no_sync()
                if world_size() > 1 and micro_step < gradient_accumulation_steps - 1
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.precision == "bf16",
                ):
                    logits = model(inputs)
                loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
                # DP averages over ranks; only divide by accumulation here.
                (loss / gradient_accumulation_steps).backward()
            step_loss += loss.detach()
            step_tokens += targets.numel()
            del logits, loss
        if world_size() > 1:
            model.finish_grad_sync()
        optimizer.step()
        # Accelerator operations are asynchronous; time completed work.
        synchronize(device)
        duration = time.perf_counter() - iteration_started

        if world_size() > 1:
            dist.all_reduce(step_loss)
            slowest = torch.tensor(duration, device=device)
            dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
            duration = slowest.item()
        final_loss = (step_loss / (gradient_accumulation_steps * world_size())).item()
        step_tokens *= world_size()
        progress["steps"] += 1
        progress["tokens_seen"] += step_tokens
        steps = progress["steps"]
        iteration_seconds += duration
        interval_seconds += duration
        interval_steps += 1
        interval_tokens += step_tokens
        if first_loss is None:
            first_loss = final_loss
        if (
            steps == initial_steps + 1
            or steps % config.log_interval == 0
            or steps == config.max_steps
        ):
            iteration_ms = 1000 * interval_seconds / interval_steps
            tokens_per_second = interval_tokens / interval_seconds
            if rank() == 0:
                print(
                    f"step={steps:04d} loss={final_loss:.6f} "
                    f"iteration_time_ms={iteration_ms:.2f} "
                    f"tokens_per_second={tokens_per_second:.1f}",
                    flush=True,
                )
            if writer is not None:
                grad_norm = (
                    torch.stack(
                        [parameter.grad.norm() for parameter in model.parameters()]
                    )
                    .norm()
                    .item()
                )
                for name, value in {
                    "train/loss": final_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    "train/tokens_seen": progress["tokens_seen"],
                    "perf/tokens_per_second": tokens_per_second,
                    "perf/iteration_time_ms": iteration_ms,
                }.items():
                    writer.add_scalar(name, value, steps)
            interval_seconds = 0.0
            interval_steps = 0
            interval_tokens = 0
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and steps % checkpoint_interval == 0
        ):
            save_checkpoint(checkpoint_path, model, optimizer, dataset, progress)
            last_saved_step = steps

    if checkpoint_path is not None and last_saved_step != progress["steps"]:
        save_checkpoint(checkpoint_path, model, optimizer, dataset, progress)
    synchronize(device)
    elapsed = time.perf_counter() - started
    steps_this_run = progress["steps"] - initial_steps
    tokens_this_run = progress["tokens_seen"] - initial_tokens
    return {
        **progress,
        "steps_this_run": steps_this_run,
        "tokens_this_run": tokens_this_run,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_this_run / elapsed,
        "iteration_time_ms": (
            1000 * iteration_seconds / steps_this_run if steps_this_run else 0.0
        ),
    }
