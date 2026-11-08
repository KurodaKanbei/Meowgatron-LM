from collections import Counter

import pytest
import torch
from torch.nn import functional as F

from core.config import ModelConfig
from core.distributed import GradientBucket
from core.model import Meow, TransformerBlock


@pytest.mark.parametrize(
    "recompute,modules",
    [
        ("full", []),
        ("selective", ["attention"]),
        ("selective", ["norm"]),
        ("selective", ["mlp_act"]),
        ("selective", ["mlp"]),
        ("selective", ["attention", "norm", "mlp_act"]),
        ("selective", ["attention", "norm", "mlp", "mlp_act"]),
    ],
)
def test_recompute_matches_forward_and_gradients(recompute, modules):
    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_heads=4,
        mlp_hidden_size=64,
        max_seq_len=8,
        num_layers=2,
    )
    reference = Meow(config)
    model = Meow(config, recompute, modules)
    model.load_state_dict(reference.state_dict())
    bucket = GradientBucket(list(model.parameters()))
    hook_calls = Counter()

    def gradient_ready(parameter):
        hook_calls[parameter] += 1

    for parameter in model.parameters():
        parameter.register_post_accumulate_grad_hook(gradient_ready)
    for _ in range(3):
        tokens = torch.randint(config.vocab_size, (2, config.max_seq_len + 1))
        actual = model(tokens[:, :-1])
        expected = reference(tokens[:, :-1])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for output in (actual, expected):
            loss = F.cross_entropy(output.flatten(0, 1), tokens[:, 1:].flatten())
            (loss / 3).backward()
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, atol=0, rtol=0
        )
        assert hook_calls[parameter] == 3
        assert (
            parameter.grad.untyped_storage().data_ptr()
            == bucket.buffer.untyped_storage().data_ptr()
        )


def test_recompute_reduces_saved_activations():
    config = ModelConfig(
        hidden_size=32, num_heads=4, mlp_hidden_size=128, max_seq_len=64
    )
    saved_bytes = {}
    for name, recompute, modules in (
        ("none", "none", []),
        ("attention", "selective", ["attention"]),
        ("norm", "selective", ["norm"]),
        ("mlp_act", "selective", ["mlp_act"]),
        ("mlp", "selective", ["mlp"]),
        ("both", "selective", ["attention", "mlp_act"]),
        ("with_norm", "selective", ["attention", "norm", "mlp_act"]),
        ("full", "full", []),
    ):
        block = TransformerBlock(config, recompute, modules)
        hidden = torch.randn(2, 64, 32, requires_grad=True)
        parameter_storages = {
            parameter.untyped_storage().data_ptr() for parameter in block.parameters()
        }
        activations = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameter_storages:
                activations[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = block(hidden)
        saved_bytes[name] = sum(activations.values())
        output.sum().backward()

    for module in ("attention", "mlp_act"):
        assert saved_bytes["full"] < saved_bytes["both"] < saved_bytes[module]
        assert saved_bytes[module] < saved_bytes["none"]
    assert saved_bytes["norm"] < saved_bytes["none"]
    assert saved_bytes["with_norm"] < saved_bytes["both"]
    assert saved_bytes["mlp"] < saved_bytes["mlp_act"]
