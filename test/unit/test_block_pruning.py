"""Block pruning (whole attention heads), compression/block_pruning.py.

Soft (zeroed o_proj columns) == hard (rows/columns removed, HF head bookkeeping
updated) == the converted decoder with the reduced head count.
"""
from __future__ import annotations

import pytest
import torch

import transformersurgeon as T
from transformersurgeon.export.common import LLMWrapper
from transformersurgeon.utils import convert_for_export


def _tiny(family):
    torch.manual_seed(0)
    kw = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
              num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=64,
              tie_word_embeddings=False)
    if family == "qwen2":
        m = T.Qwen2ForCausalLMCompress(T.Qwen2ConfigCompress(**kw))
        mgr = T.models.qwen2_c.Qwen2CompressionSchemesManager(m)
    else:
        m = T.LlamaForCausalLMCompress(T.LlamaConfigCompress(**kw))
        mgr = T.models.llama_c.LlamaCompressionSchemesManager(m)
    return m.eval(), mgr


IDS = torch.arange(1, 25) * 7 % 256


def _logits(m):
    with torch.no_grad():
        return m(IDS[None]).logits[0]


# layer 0: heads {1, 6} (one per KV group of 4); layer 1: heads {0, 3, 4, 5} (two per group)
REMOVED = {0: [1, 6], 1: [0, 3, 4, 5]}


@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_soft_equals_hard_equals_converted(family):
    m, mgr = _tiny(family)
    for i, heads in REMOVED.items():
        mgr.set("block_pruning", "blocks", heads, criteria=[["self_attn.q_proj", i]])
    mgr.apply(hard=False)
    soft = _logits(m)
    mgr.apply(hard=True)
    hard = _logits(m)
    torch.testing.assert_close(hard, soft, rtol=1e-5, atol=1e-5)

    for i, heads in REMOVED.items():
        attn = m.model.layers[i].self_attn
        kept = 8 - len(heads)
        assert attn.q_proj.out_features == kept * 8 and attn.o_proj.in_features == kept * 8
        assert attn.num_key_value_groups == kept // 2
        assert attn.k_proj.out_features == 2 * 8  # KV heads untouched

    n = len(IDS)
    dec = convert_for_export(m, options={"use_sdpa": False, "max_cache_len": n})["text"]
    assert [b.attn.num_heads for b in dec.blocks] == [6, 4]
    w = LLMWrapper(m.get_input_embeddings(), dec, m.lm_head).eval()
    with torch.no_grad():
        conv = torch.stack([w(IDS[j:j + 1], torch.tensor([j])).reshape(-1) for j in range(n)])
    torch.testing.assert_close(conv, hard, rtol=1e-4, atol=1e-4)


def test_soft_restore_and_head_removal_changes_output():
    m, mgr = _tiny("qwen2")
    base = _logits(m)
    mgr.set("block_pruning", "blocks", [2, 5], criteria=[["self_attn.q_proj", 0]])
    mgr.apply(hard=False)
    assert not torch.allclose(_logits(m), base)
    mgr.restore()
    torch.testing.assert_close(_logits(m), base)


@pytest.mark.parametrize("method", ["magnitude", "random"])
def test_ratio_removes_uniformly_per_group(method):
    m, mgr = _tiny("llama")
    mgr.set("block_pruning", "ratio", 0.5, criteria="self_attn.q_proj")
    mgr.set("block_pruning", "method", method, criteria="self_attn.q_proj")
    mgr.apply(hard=True)
    for layer in m.model.layers:
        kept = layer.self_attn.q_proj.block_prune_mask
        assert kept.view(2, 4).sum(dim=1).tolist() == [2, 2]
        assert layer.self_attn.num_key_value_groups == 2
    assert torch.isfinite(_logits(m)).all()


def test_rejects_uneven_groups_and_other_layers():
    m, mgr = _tiny("qwen2")
    mgr.set("block_pruning", "blocks", [0, 1], criteria=[["self_attn.q_proj", 0]])  # both from group 0
    with pytest.raises(ValueError, match="same number of blocks from every unit"):
        mgr.apply(hard=True)

    m, mgr = _tiny("qwen2")
    mgr.set("block_pruning", "blocks", [0], criteria=[["self_attn.k_proj", 0]])
    with pytest.raises(ValueError, match="not defined"):
        mgr.apply(hard=True)
