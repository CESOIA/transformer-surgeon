"""Per-kv-group prunable RoPE + pruned KV-cache (Parts B & C).

Exercises hard-pruned GQA q/k where different kv-groups keep *different* rotary
frequencies -- previously rejected by the uniform-across-all-heads assumption.
"""
from __future__ import annotations

import pytest
import torch

from transformersurgeon.blocks.mha import MHACausal, MHAEncoder
from transformersurgeon.blocks.rope import (
    build_rope_prune_projection,
    _derive_rope_prune_pattern,
    precompute_rope_cos_sin_half,
    precompute_rope_inv_freqs,
)

HEAD_DIM = 8
HALF = HEAD_DIM // 2


def _head_mask(freqs):
    """Full head_dim keep-mask keeping rotary frequencies ``freqs`` (real+imag)."""
    m = torch.zeros(HEAD_DIM, dtype=torch.bool)
    for f in freqs:
        m[f] = True
        m[f + HALF] = True
    return m


def _distinct_gqa_masks():
    # kv-group0 keeps freqs {0,1}; group1 keeps {1,2}. Same count, real==imag.
    g0, g1 = _head_mask([0, 1]), _head_mask([1, 2])
    q_keep = torch.cat([g0, g0, g1, g1])  # 4 q heads (group_size 2)
    k_keep = torch.cat([g0, g1])          # 2 kv heads
    return q_keep, k_keep, g0, g1


# --- Part B: per-kv-group projection --------------------------------------


def test_derive_per_kv_group_patterns_distinct():
    q_keep, k_keep, _, _ = _distinct_gqa_masks()
    q = _derive_rope_prune_pattern(q_keep, HEAD_DIM, 4, num_kv_groups=2, source="q")
    k = _derive_rope_prune_pattern(k_keep, HEAD_DIM, 2, num_kv_groups=2, source="k")
    assert q.shape == k.shape == (2, HALF)
    assert torch.equal(q, k)
    assert not torch.equal(q[0], q[1])  # the two kv-groups differ


def test_build_projection_selects_per_group_frequencies():
    q_keep, k_keep, _, _ = _distinct_gqa_masks()
    proj = build_rope_prune_projection(q_keep, k_keep, HEAD_DIM, 4, 2)
    assert proj.shape == (2, 2, HALF)  # (kv_groups, kept_freqs, head_dim//2)
    assert proj[0].argmax(-1).tolist() == [0, 1]
    assert proj[1].argmax(-1).tolist() == [1, 2]


def test_projection_rejects_split_real_imag_pair():
    bad = torch.zeros(HEAD_DIM, dtype=torch.bool)
    bad[0] = True  # freq0 real kept, imag pruned
    with pytest.raises(ValueError):
        _derive_rope_prune_pattern(torch.cat([bad, bad]), HEAD_DIM, 2, num_kv_groups=2, source="k")


def test_projection_rejects_heads_disagreeing_within_group():
    _, _, g0, g1 = _distinct_gqa_masks()
    # group0's two heads disagree (g0 then g1).
    with pytest.raises(ValueError):
        _derive_rope_prune_pattern(torch.cat([g0, g1, g1, g1]), HEAD_DIM, 4, num_kv_groups=2, source="q")


def test_projection_rejects_unequal_counts_across_groups():
    g0, g2 = _head_mask([0, 1]), _head_mask([0])  # counts 2 vs 1
    with pytest.raises(ValueError):
        _derive_rope_prune_pattern(torch.cat([g0, g2]), HEAD_DIM, 2, num_kv_groups=2, source="k")


def test_projection_rejects_q_k_pattern_mismatch():
    q_keep, _, g0, g1 = _distinct_gqa_masks()
    k_swapped = torch.cat([g1, g0])  # k groups keep the opposite frequencies
    with pytest.raises(ValueError):
        build_rope_prune_projection(q_keep, k_swapped, HEAD_DIM, 4, 2)


# --- Part B/C: pruned MHA forward -----------------------------------------


def _hard_prune_qk(mod, q_keep, k_keep, num_heads, kv_num_heads):
    with torch.no_grad():
        mod.q_proj.weight = torch.nn.Parameter(mod.q_proj.weight[q_keep, :].clone())
        mod.k_proj.weight = torch.nn.Parameter(mod.k_proj.weight[k_keep, :].clone())
    mod.q_proj.out_features = int(q_keep.sum())
    mod.k_proj.out_features = int(k_keep.sum())
    mod.q_proj.register_buffer("rope_prune_mask", q_keep)
    mod.k_proj.register_buffer("rope_prune_mask", k_keep)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_encoder_forward_with_per_group_pruned_rope(use_sdpa):
    q_keep, k_keep, _, _ = _distinct_gqa_masks()
    m = MHAEncoder(32, 4, kv_num_heads=2, use_sdpa=use_sdpa).eval()
    _hard_prune_qk(m, q_keep, k_keep, 4, 2)
    inv = precompute_rope_inv_freqs(head_dim=HEAD_DIM, base=1e4)
    cos, sin = precompute_rope_cos_sin_half(inv, torch.tensor(5), torch.tensor(0))
    out = m(torch.randn(5, 32), rope=(cos, sin))
    assert out.shape == (5, 32)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cache_impl", ["mutable", "io_scatter", "io_concat"])
def test_causal_pruned_kv_cache_dims(cache_impl):
    q_keep, k_keep, _, _ = _distinct_gqa_masks()
    m = MHACausal(32, 4, kv_num_heads=2, use_sdpa=False, cache_impl=cache_impl,
                  max_cache_len=16, dtype=torch.float32).eval()
    _hard_prune_qk(m, q_keep, k_keep, 4, 2)

    # Key cache shrinks to the pruned key head_dim; value cache keeps head_dim.
    assert m.key_head_dim == 4
    assert m.value_head_dim == HEAD_DIM

    inv = precompute_rope_inv_freqs(head_dim=HEAD_DIM, base=1e4)
    cos, sin = precompute_rope_cos_sin_half(inv, torch.tensor(16), torch.tensor(0))
    q_pos = k_pos = torch.arange(16)
    mask_penalty = torch.full((16, 16), float("-inf"))
    kc = torch.zeros(m.max_cache_length, m.kv_num_heads, m.key_head_dim)
    vc = torch.zeros(m.max_cache_length, m.kv_num_heads, m.value_head_dim)

    for pos in range(3):
        pid = torch.tensor([pos])
        args = (torch.randn(1, 32), pid, (q_pos, k_pos), mask_penalty)
        if cache_impl == "mutable":
            out = m(*args, rope=(cos, sin))
        else:
            out, kc, vc = m(*args, key_cache=kc, value_cache=vc, rope=(cos, sin))
        assert torch.isfinite(out).all()

    assert m.key_cache.shape[-1] == 4
    assert m.value_cache.shape[-1] == HEAD_DIM
