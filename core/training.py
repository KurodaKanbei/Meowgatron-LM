"""A single-device training loop with native cross-entropy and AdamW."""

import time
from itertools import islice

import torch
from torch.nn import functional as F

from core.config import TrainConfig
from core.device import synchronize


def train(
    model,
    dataloader,
    optimizer,
    config: TrainConfig,
    device: torch.device,
    writer=None,
    checkpoint_path=None,
    gradient_accumulation_steps=1,
):
    model.train()
    synchronize(device)
    started = time.perf_counter()
    tokens_seen = 0
    first_loss = None
    final_loss = None
    iteration_seconds = 0.0
    interval_seconds = 0.0
    interval_steps = 0
    interval_tokens = 0

    batches = zip(*[iter(dataloader)] * gradient_accumulation_steps)
    # Start before iteration so timing includes retrieving every microbatch.
    iteration_started = time.perf_counter()
    for steps, microbatches in enumerate(islice(batches, config.max_steps), 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        for inputs, targets in microbatches:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.precision == "bf16",
            ):
                logits = model(inputs)
            # Loss and softmax reductions use FP32, including in BF16 runs.
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
            (loss / gradient_accumulation_steps).backward()
            step_loss += loss.detach()
            step_tokens += targets.numel()
            del logits, loss
        optimizer.step()
        # Accelerator operations are asynchronous; time completed work.
        synchronize(device)
        duration = time.perf_counter() - iteration_started

        final_loss = (step_loss / gradient_accumulation_steps).item()
        tokens_seen += step_tokens
        iteration_seconds += duration
        interval_seconds += duration
        interval_steps += 1
        interval_tokens += step_tokens
        if steps == 1:
            first_loss = final_loss
        if steps == 1 or steps % config.log_interval == 0 or steps == config.max_steps:
            iteration_ms = 1000 * interval_seconds / interval_steps
            tokens_per_second = interval_tokens / interval_seconds
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
                    "train/tokens_seen": tokens_seen,
                    "perf/tokens_per_second": tokens_per_second,
                    "perf/iteration_time_ms": iteration_ms,
                }.items():
                    writer.add_scalar(name, value, steps)
            interval_seconds = 0.0
            interval_steps = 0
            interval_tokens = 0
        if (
            config.checkpoint_interval is not None
            and steps % config.checkpoint_interval == 0
        ):
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": steps,
                    "tokens_seen": tokens_seen,
                },
                temporary_path,
            )
            temporary_path.replace(checkpoint_path)
        # Metrics and checkpoint writes are outside the next iteration's timing.
        iteration_started = time.perf_counter()

    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "tokens_seen": tokens_seen,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_seen / elapsed,
        "iteration_time_ms": 1000 * iteration_seconds / steps,
    }
