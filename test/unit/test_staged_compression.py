"""Staged hard compression: after a hard apply, a compression type not applied yet can be set and hard-applied on
top of it (pruning -> LRD -> quantization in separate apply() calls), while the applied ones stay locked."""
from __future__ import annotations

import pytest
import torch

import transformersurgeon as T


def _model():
    torch.manual_seed(0)
    cfg = T.Qwen2ConfigCompress(vocab_size=256, hidden_size=64, intermediate_size=256, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    m = T.Qwen2ForCausalLMCompress(cfg).eval()
    return m, T.models.qwen2_c.Qwen2CompressionSchemesManager(m)


def _quant(mgr, crit):
    for k, v in (("precision", 4), ("granularity", "per_group"), ("group_size", 32)):
        mgr.set("quantization", k, v, criteria=crit)


def test_lrd_then_quantization_on_the_same_layer():
    m, mgr = _model()
    mgr.set("lrd", "rank", 32, criteria=[["mlp.up_proj", 0]])
    mgr.apply(hard=True, criteria=["mlp.up_proj"])
    up = m.model.layers[0].mlp.up_proj
    assert up.linear_V.weight.shape == (32, 64) and up.weight.shape == (256, 32)
    _quant(mgr, [["mlp.up_proj", 0]])
    mgr.apply(hard=True, criteria=["mlp.up_proj"])
    assert hasattr(up, "_torchao_scale")  # quantized on top of the LRD factors
    assert not hasattr(m.model.layers[1].mlp.up_proj, "_torchao_scale")
    out = m(torch.randint(0, 256, (1, 8))).logits
    assert torch.isfinite(out).all()


def test_pruning_then_quantization_on_the_same_layer():
    m, mgr = _model()
    for g, paths in mgr.auto_groups().items():
        if any("gate_proj" in p for p in paths):
            mgr.set("structured_pruning", "share_mask", True, group=g)
            mgr.set("structured_pruning", "reduce_op", "add", group=g)
    for n in ("mlp.gate_proj", "mlp.up_proj"):
        mgr.set("structured_pruning", "ratio", 0.5, criteria=[[n, 0]])
    mgr.apply(hard=True, criteria=["mlp.gate_proj", "mlp.up_proj"])
    mlp = m.model.layers[0].mlp
    assert mlp.up_proj.out_features == 128 and mlp.down_proj.in_features == 128
    _quant(mgr, [[n, 0] for n in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")])
    mgr.apply(hard=True, criteria=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    assert all(hasattr(getattr(mlp, n), "_torchao_scale") for n in ("gate_proj", "up_proj", "down_proj"))
    assert mlp.down_proj.in_features == 128
    assert torch.isfinite(m(torch.randint(0, 256, (1, 8))).logits).all()


def test_applied_compression_stays_locked():
    m, mgr = _model()
    mgr.set("lrd", "rank", 32, criteria=[["mlp.up_proj", 0]])
    mgr.apply(hard=True, criteria=["mlp.up_proj"])
    with pytest.raises(RuntimeError, match="already been applied"):
        mgr.set("lrd", "rank", 16, criteria=[["mlp.up_proj", 0]])
    _quant(mgr, [["mlp.up_proj", 0]])
    with pytest.raises(RuntimeError, match="must be hard"):
        mgr.apply(hard=False, criteria=["mlp.up_proj"])
