"""LPBQ-style group scales (quantization group_scale_bits), Qualcomm HTP's block format.

Group scales must be channel_scale * m with an integer m in [1, 2**bits]; the QNN
builder re-derives exactly that encoding (channel scale = max / 2**bits), so the
torch model and the HTP weights are identical.
"""
from __future__ import annotations

import pytest
import torch

import transformersurgeon as T
from transformersurgeon.compression.quantization_methods.vanilla import lpbq_group_scales, quantize_vanilla


def test_lpbq_scales_are_integer_multiples_of_channel_scale():
    torch.manual_seed(0)
    w = torch.randn(64, 512) * torch.rand(64, 1) * torch.linspace(0.1, 3, 512)
    q, scale, deq = quantize_vanilla(w, 4, granularity="per_group", group_size=32, group_scale_bits=4)
    channel = scale.amax(dim=1, keepdim=True) / 16
    m = scale / channel
    torch.testing.assert_close(m, m.round(), rtol=0, atol=1e-4)
    assert m.min() >= 1 and m.max() <= 16 and torch.all(m.amax(dim=1) == 16)
    # what the QNN builder derives from these scales (node_visitor.make_qnn_per_block_config)
    q_scales = torch.clamp(torch.round(scale / channel), 1, 16)
    torch.testing.assert_close(channel * q_scales, scale)
    # no group's range is clipped: its scale is >= the free max-abs scale
    free = w.reshape(64, 16, 32).abs().amax(-1) / 7
    assert torch.all(scale >= free * (1 - 1e-5))
    assert q.abs().max() <= 7
    err = (deq(q, scale) - w).abs().max()
    assert err <= scale.max() / 2 + 1e-6


def test_zero_rows_stay_finite():
    s = lpbq_group_scales(torch.zeros(2, 4), 4)
    assert torch.all(s == 0)


def test_manager_hard_lpbq_and_validation():
    torch.manual_seed(0)
    cfg = lambda: T.Qwen2ConfigCompress(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                                        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    m = T.Qwen2ForCausalLMCompress(cfg()).eval()
    mgr = T.models.qwen2_c.Qwen2CompressionSchemesManager(m)
    for k, v in (("precision", 4), ("granularity", "per_group"), ("group_size", 32), ("group_scale_bits", 4)):
        mgr.set("quantization", k, v, criteria="mlp")
    mgr.apply(hard=True, criteria="mlp")
    up = m.model.layers[0].mlp.up_proj
    assert up._torchao_group_scale_bits == 4
    s = up._torchao_scale.float()
    ratio = s / (s.amax(dim=1, keepdim=True) / 16)
    torch.testing.assert_close(ratio, ratio.round(), rtol=0, atol=1e-3)

    m = T.Qwen2ForCausalLMCompress(cfg()).eval()  # fresh config: it carries the manager's compression state
    mgr = T.models.qwen2_c.Qwen2CompressionSchemesManager(m)
    mgr.set("quantization", "precision", 4, criteria="mlp")
    mgr.set("quantization", "group_scale_bits", 4, criteria="mlp")  # per_tensor default: invalid
    with pytest.raises(NotImplementedError, match="group_scale_bits"):
        mgr.apply(hard=True, criteria="mlp")
