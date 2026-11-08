import copy

import torch
from torch.nn import functional as F

from core.attention import SelfAttention
from core.config import ModelConfig


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
