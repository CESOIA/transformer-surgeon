"""End-to-end GQA attention pruning on a tiny Qwen2: auto per-kv-group q/k mask
-> hard prune -> convert -> decode. Exercises Parts A (mask), B (RoPE projection)
and C (pruned KV cache) together, offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _helpers.model_factory import FAMILIES  # noqa: E402

from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager  # noqa: E402
from transformersurgeon.utils import convert_for_export  # noqa: E402
from transformersurgeon.export.common import LLMWrapper, build_zero_caches  # noqa: E402

pytestmark = pytest.mark.e2e

HEAD_DIM = 16  # tiny qwen2: hidden_size 64 / num_attention_heads 4
HALF = HEAD_DIM // 2


def _configure_qk_auto(mgr, ratio=0.25):
    groups = mgr.auto_groups()
    qk = [g for g, paths in groups.items() if any("q_proj" in p for p in paths)]
    assert qk, "no q/k coupled group found"
    for g in groups:
        mgr.set("structured_pruning", "share_mask", True, group=g)
        mgr.set("structured_pruning", "reduce_op", "add", group=g)
    for g in qk:
        mgr.set("structured_pruning", "method", "magnitude", group=g)
        mgr.set("structured_pruning", "granularity", HEAD_DIM, group=g)
        mgr.set("structured_pruning", "repeated_pattern", "auto", group=g)
        mgr.set("structured_pruning", "ratio", ratio, group=g)
    return qk


def test_auto_qk_soft_mask_layout():
    model = FAMILIES["qwen2"].build().eval()
    mgr = Qwen2CompressionSchemesManager(model)
    _configure_qk_auto(mgr)
    mgr.apply(hard=False)

    attn = model.model.layers[0].self_attn
    qv = attn.q_proj.weight_mask.view(4, HEAD_DIM)  # 4 q heads
    kv = attn.k_proj.weight_mask.view(2, HEAD_DIM)   # 2 kv heads, group_size 2
    # Each kv-group (2 q heads + its k head) shares one pattern; groups differ.
    assert torch.equal(qv[0], qv[1]) and torch.equal(qv[0], kv[0])
    assert torch.equal(qv[2], qv[3]) and torch.equal(qv[2], kv[1])
    assert not torch.equal(qv[0], qv[2])
    # RoPE-valid: real/imag paired within every head.
    for i in range(4):
        assert torch.equal(qv[i][:HALF], qv[i][HALF:])


def test_auto_qk_hard_prune_convert_decode():
    model = FAMILIES["qwen2"].build().eval()
    mgr = Qwen2CompressionSchemesManager(model)
    _configure_qk_auto(mgr)
    mgr.apply(hard=True)
    mgr.prepare_for_save()

    attn = model.model.layers[0].self_attn
    # 2 of 8 rotary freqs pruned per head -> 4 channels/head removed.
    assert attn.q_proj.out_features == 4 * (HEAD_DIM - 4)   # 48
    assert attn.k_proj.out_features == 2 * (HEAD_DIM - 4)   # 24
    assert attn.v_proj.out_features == 2 * HEAD_DIM         # v unpruned

    converted = convert_for_export(
        model, options={"use_sdpa": False, "cache_impl": "mutable", "max_cache_len": 32}
    )
    wrapper = LLMWrapper(model.get_input_embeddings(), converted["text"], model.lm_head).eval()

    cattn = wrapper.decoder.blocks[0].attn
    assert cattn.key_head_dim == HEAD_DIM - 4   # 12, pruned
    assert cattn.value_head_dim == HEAD_DIM     # 16, unpruned

    with torch.no_grad():
        logits = None
        for pos in range(5):
            logits = wrapper(torch.tensor([pos % 50 + 1]), torch.tensor([pos]))
    assert torch.isfinite(logits).all()


def test_vproj_prune_cascades_to_oproj_via_coupled_repeated_pattern():
    """v_proj output pruning cascades to o_proj's input under GQA: each kv head's
    kept value channels are repeated group_size times (coupled_repeated_pattern)
    to match attention's repeat_interleave expansion before o_proj."""
    model = FAMILIES["qwen2"].build().eval()
    mgr = Qwen2CompressionSchemesManager(model)
    mgr.auto_groups()
    # granularity=head_dim so each kv head is one cascade chunk; repeat=group_size=2.
    mgr.set("structured_pruning", "method", "magnitude", criteria="v_proj")
    mgr.set("structured_pruning", "granularity", HEAD_DIM, criteria="v_proj")
    mgr.set("structured_pruning", "coupled_repeated_pattern", 2, criteria="v_proj")
    mgr.set("structured_pruning", "ratio", 0.25, criteria="v_proj")
    mgr.apply(hard=True)
    mgr.prepare_for_save()

    attn = model.model.layers[0].self_attn
    v_head_dim = HEAD_DIM - 4  # 12 kept per kv head
    assert attn.v_proj.out_features == 2 * v_head_dim          # 24 (2 kv heads)
    assert attn.o_proj.weight.shape[1] == 4 * v_head_dim       # 48 (4 q heads)

    converted = convert_for_export(
        model, options={"use_sdpa": False, "cache_impl": "mutable", "max_cache_len": 32}
    )
    wrapper = LLMWrapper(model.get_input_embeddings(), converted["text"], model.lm_head).eval()
    cattn = wrapper.decoder.blocks[0].attn
    assert cattn.value_head_dim == v_head_dim
    assert cattn.out_proj.in_features == 4 * v_head_dim

    with torch.no_grad():
        logits = None
        for pos in range(5):
            logits = wrapper(torch.tensor([pos % 40 + 1]), torch.tensor([pos]))
    assert torch.isfinite(logits).all()


def test_auto_qk_hard_prune_io_cache_sizes():
    model = FAMILIES["qwen2"].build().eval()
    mgr = Qwen2CompressionSchemesManager(model)
    _configure_qk_auto(mgr)
    mgr.apply(hard=True)
    mgr.prepare_for_save()

    converted = convert_for_export(
        model, options={"use_sdpa": False, "cache_impl": "io_scatter", "max_cache_len": 32}
    )
    key_caches, value_caches = build_zero_caches(converted["text"])
    assert key_caches and value_caches
    assert key_caches[0].shape[-1] == HEAD_DIM - 4  # pruned key cache
    assert value_caches[0].shape[-1] == HEAD_DIM    # full value cache
