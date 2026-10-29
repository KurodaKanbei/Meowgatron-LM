import copy

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.config import ModelConfig, TrainConfig
from core.data import TokenDataset
from core.model import Meow
from core.training import train


def test_text_packing_preserves_document_boundaries_and_target_shift(tmp_path):
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "<|endoftext|>": 1, "a": 2, "b": 3, "c": 4, "d": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    pq.write_table(pa.table({"text": ["a b c", "d"]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"text": ["a b c d"]}), tmp_path / "b.parquet")
    dataset = TokenDataset(
        [tmp_path / "b.parquet", tmp_path / "a.parquet"], tokenizer, seq_len=3
    )
    samples = list(dataset)
    stream = torch.tensor([2, 3, 4, 1, 5, 1, 2, 3, 4, 5, 1])
    assert len(samples) == 3
    torch.testing.assert_close(torch.cat([x for x, _ in samples]), stream[:9])
    torch.testing.assert_close(torch.cat([y for _, y in samples]), stream[1:10])
    for (x, y), (again_x, again_y) in zip(samples, list(dataset)):
        torch.testing.assert_close(x, again_x)
        torch.testing.assert_close(y, again_y)


def small_model():
    return Meow(
        ModelConfig(
            vocab_size=8,
            hidden_size=24,
            num_heads=2,
            mlp_hidden_size=48,
            max_seq_len=8,
            num_layers=2,
        )
    )


def optimizer_for(model):
    return torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.0, foreach=False, fused=False
    )


def test_native_training_learns_a_repeating_token_sequence(tmp_path):
    torch.manual_seed(42)
    model = small_model()
    optimizer = optimizer_for(model)
    inputs = (torch.arange(8).repeat(162, 1) + torch.arange(162)[:, None]) % 8
    targets = (inputs + 1) % 8
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    result = train(
        model,
        loader,
        optimizer,
        TrainConfig(max_steps=None, log_interval=40, checkpoint_interval=20),
        torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt",
        gradient_accumulation_steps=2,
    )
    assert result["steps"] == 40
    assert result["final_loss"] < 0.2 * result["first_loss"]
    assert result["tokens_seen"] == 40 * 4 * 8
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["steps"] == 40
    assert checkpoint["tokens_seen"] == result["tokens_seen"]
    torch.testing.assert_close(checkpoint["model"], model.state_dict())
    assert len(checkpoint["optimizer"]["state"]) == len(list(model.parameters()))
    for state in checkpoint["optimizer"]["state"].values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32

    reference = small_model()
    accumulated = copy.deepcopy(reference)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.01)
    reference_loss = F.cross_entropy(
        reference(inputs[:4]).flatten(0, 1), targets[:4].flatten()
    )
    reference_loss.backward()
    reference_optimizer.step()
    result = train(
        accumulated,
        [(inputs[:2], targets[:2]), (inputs[2:4], targets[2:4])],
        accumulated_optimizer,
        TrainConfig(max_steps=1),
        torch.device("cpu"),
        gradient_accumulation_steps=2,
    )
    assert abs(result["first_loss"] - reference_loss.item()) < 1e-6
    for parameter, expected in zip(accumulated.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-7)


def test_bf16_uses_fp32_parameters_gradients_and_optimizer_state():
    torch.manual_seed(23)
    reference = small_model()
    model = copy.deepcopy(reference)
    inputs = torch.arange(8).repeat(2, 1)
    targets = (inputs + 1) % 8
    reference_loss = F.cross_entropy(reference(inputs).flatten(0, 1), targets.flatten())
    reference_loss.backward()
    reference_gradients = torch.cat(
        [parameter.grad.flatten() for parameter in reference.parameters()]
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model(inputs).dtype == torch.bfloat16
    optimizer = optimizer_for(model)
    result = train(
        model,
        [(inputs, targets)] * 2,
        optimizer,
        TrainConfig(max_steps=1, precision="bf16"),
        torch.device("cpu"),
    )
    assert result["steps"] == 1
    assert abs(result["first_loss"] - reference_loss.item()) < 0.01
    gradients = torch.cat(
        [parameter.grad.flatten() for parameter in model.parameters()]
    )
    relative_error = (
        gradients - reference_gradients
    ).norm() / reference_gradients.norm()
    assert relative_error < 0.03
    for parameter in model.parameters():
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
        state = optimizer.state[parameter]
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
