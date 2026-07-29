"""attn_impl='custom_sdpa' on MHACausal: numeric parity vs 'manual', and the
construction-/forward-time validation guards it adds.

Requires ExecuTorch's LLM custom-ops extension (torch.ops.llama.custom_sdpa /
update_cache) -- skipped when unavailable.
"""
from __future__ import annotations

import pytest
import torch

from _helpers import capabilities as caps

from transformersurgeon.blocks.mha import MHACausal
from transformersurgeon.blocks.rope import (
    precompute_rope_cos_sin_half,
    precompute_rope_inv_freqs,
)

pytestmark = caps.requires_custom_sdpa

HEAD_DIM = 8
HALF = HEAD_DIM // 2


def _head_mask(freqs):
    m = torch.zeros(HEAD_DIM, dtype=torch.bool)
    for f in freqs:
        m[f] = True
        m[f + HALF] = True
    return m


def _hard_prune_qk(mod, q_keep, k_keep):
    with torch.no_grad():
        mod.q_proj.weight = torch.nn.Parameter(mod.q_proj.weight[q_keep, :].clone())
        mod.k_proj.weight = torch.nn.Parameter(mod.k_proj.weight[k_keep, :].clone())
    mod.q_proj.out_features = int(q_keep.sum())
    mod.k_proj.out_features = int(k_keep.sum())
    mod.q_proj.register_buffer("rope_prune_mask", q_keep)
    mod.k_proj.register_buffer("rope_prune_mask", k_keep)
    mod.finalize_rope_pruning()


def test_causal_custom_sdpa_matches_manual():
    torch.manual_seed(0)
    kwargs = dict(kv_num_heads=2, max_cache_len=16, cache_impl="mutable", dtype=torch.float32)
    m_manual = MHACausal(32, 4, attn_impl="manual", **kwargs).eval()
    m_custom = MHACausal(32, 4, attn_impl="custom_sdpa", **kwargs).eval()
    # persistent=False buffers (key_cache/value_cache/custom_sdpa_*_cache) are
    # excluded from state_dict -- this only copies q/k/v/out_proj weights.
    m_custom.load_state_dict(m_manual.state_dict())

    q_pos = k_pos = torch.arange(16)
    mask_penalty = torch.full((16, 16), float("-inf"))

    for pos in range(5):
        pid = torch.tensor([pos])
        x = torch.randn(1, 32)
        attn_mask = torch.where((q_pos < k_pos), mask_penalty, torch.zeros_like(mask_penalty))[pid].unsqueeze(0)
        out_manual = m_manual(x, pid, attn_mask)
        out_custom = m_custom(x, pid, attn_mask)
        torch.testing.assert_close(out_manual, out_custom, atol=1e-4, rtol=1e-3)


def test_custom_sdpa_requires_mutable_cache_impl():
    with pytest.raises(ValueError, match="cache_impl='mutable'"):
        MHACausal(32, 4, kv_num_heads=2, max_cache_len=16, cache_impl="io_scatter",
                  dtype=torch.float32, attn_impl="custom_sdpa")


def test_custom_sdpa_rejects_unknown_attn_impl():
    with pytest.raises(ValueError, match="Unsupported attn_impl"):
        MHACausal(32, 4, kv_num_heads=2, max_cache_len=16, dtype=torch.float32, attn_impl="bogus")


def test_custom_sdpa_rejects_pruned_q_head_dim_mismatch():
    """RoPE-linked structured pruning can shrink q/k's head_dim while v_proj
    stays unpruned -- torch.ops.llama.custom_sdpa sizes its output like
    `query`, not `value`, so that combination must raise rather than silently
    produce wrong-shaped output.
    """
    g0, g1 = _head_mask([0, 1]), _head_mask([1, 2])
    q_keep = torch.cat([g0, g0, g1, g1])
    k_keep = torch.cat([g0, g1])

    m = MHACausal(32, 4, kv_num_heads=2, cache_impl="mutable", max_cache_len=16,
                  dtype=torch.float32, attn_impl="custom_sdpa").eval()
    _hard_prune_qk(m, q_keep, k_keep)
    assert m.key_head_dim == 4
    assert m.value_head_dim == HEAD_DIM

    q_pos = k_pos = torch.arange(16)
    mask_penalty = torch.full((16, 16), float("-inf"))
    pid = torch.tensor([0])
    attn_mask = torch.where((q_pos < k_pos), mask_penalty, torch.zeros_like(mask_penalty))[pid].unsqueeze(0)

    with pytest.raises(RuntimeError, match="q_head_dim == value_head_dim"):
        m(torch.randn(1, 32), pid, attn_mask)
