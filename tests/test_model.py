import copy

import torch
from torch.nn import functional as F

from core.attention import SelfAttention
from core.config import ModelConfig
from core.model import Meow


def test_attention_matches_pytorch_reference_forward_and_backward():
    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=16, hidden_size=16, num_heads=4, mlp_hidden_size=32, max_seq_len=8
    )
    attention = SelfAttention(config)
    reference = copy.deepcopy(attention)
    hidden = torch.randn(2, 7, 16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    actual = attention(hidden)

    q, k, v = reference.qkv(reference_hidden).chunk(3, dim=-1)
    q, k, v = [tensor.reshape(2, 7, 4, 4).transpose(1, 2) for tensor in (q, k, v)]
    expected = reference.proj(
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        .transpose(1, 2)
        .reshape(2, 7, 16)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-6, rtol=1e-5)
    for actual_parameter, expected_parameter in zip(
        attention.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, atol=2e-6, rtol=1e-5
        )


def test_future_tokens_cannot_change_past_logits():
    torch.manual_seed(11)
    model = Meow(
        ModelConfig(
            vocab_size=16,
            hidden_size=16,
            num_heads=2,
            mlp_hidden_size=32,
            max_seq_len=8,
        )
    )
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = inputs.clone()
    changed[:, 3:] = torch.tensor([9, 10, 11])
    with torch.no_grad():
        original_logits = model(inputs)
        changed_logits = model(changed)
    torch.testing.assert_close(
        original_logits[:, :3], changed_logits[:, :3], atol=0, rtol=0
    )
    assert not torch.allclose(original_logits[:, 3:], changed_logits[:, 3:])


def test_tied_embedding_combines_input_and_output_gradients():
    torch.manual_seed(13)
    model = Meow(
        ModelConfig(
            vocab_size=16,
            hidden_size=16,
            num_heads=2,
            mlp_hidden_size=32,
            max_seq_len=8,
        )
    )
    reference = copy.deepcopy(model)
    reference.lm_head.weight = torch.nn.Parameter(
        reference.token_embedding.weight.detach().clone()
    )
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([2, 3, 4, 5])
    logits = model(inputs)
    reference_logits = reference(inputs)
    torch.testing.assert_close(logits, reference_logits)
    F.cross_entropy(logits.flatten(0, 1), targets).backward()
    F.cross_entropy(reference_logits.flatten(0, 1), targets).backward()
    expected_gradient = (
        reference.token_embedding.weight.grad + reference.lm_head.weight.grad
    )
    torch.testing.assert_close(model.token_embedding.weight.grad, expected_gradient)
    initial_weight = model.token_embedding.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    torch.testing.assert_close(
        model.lm_head.weight, initial_weight - 0.01 * expected_gradient
    )
    assert model.lm_head.weight is model.token_embedding.weight
