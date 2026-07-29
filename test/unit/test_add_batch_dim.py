"""add_batch_dim (MHACausal / TransformerDecoder): numeric parity against the
default unbatched contract, including under per-kv-group pruned RoPE, plus a
full TransformerDecoder-level shape check. See AGENTS.md's "add_batch_dim"
section and XNNPACK_DECODE_SPEED_FIX.md for why this option exists.
"""
from __future__ import annotations

import pytest
import torch

from transformersurgeon.blocks.config import CustomDecoderConfigCompress
from transformersurgeon.blocks.decoder import TransformerDecoder
from transformersurgeon.blocks.mha import MHACausal
from transformersurgeon.blocks.rope import (
    precompute_rope_cos_sin_half,
    precompute_rope_inv_freqs,
)

HEAD_DIM = 8
HALF = HEAD_DIM // 2


def _head_mask(freqs):
    m = torch.zeros(HEAD_DIM, dtype=torch.bool)
    for f in freqs:
        m[f] = True
        m[f + HALF] = True
    return m


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa", "custom_sdpa"])
def test_add_batch_dim_matches_unbatched(attn_impl):
    if attn_impl == "custom_sdpa":
        pytest.importorskip("executorch.extension.llm.custom_ops.custom_ops")

    torch.manual_seed(0)
    kwargs = dict(kv_num_heads=2, max_cache_len=16, cache_impl="mutable", dtype=torch.float32, attn_impl=attn_impl)
    m_unbatched = MHACausal(32, 4, add_batch_dim=False, **kwargs).eval()
    m_batched = MHACausal(32, 4, add_batch_dim=True, **kwargs).eval()
    m_batched.load_state_dict(m_unbatched.state_dict())

    q_pos = k_pos = torch.arange(16)
    mask_penalty = torch.full((16, 16), float("-inf"))

    for pos in range(5):
        pid = torch.tensor([pos])
        x2d = torch.randn(1, 32)
        x3d = x2d.unsqueeze(0)  # (1, 1, 32) -- batch=1
        attn_mask = torch.where((q_pos < k_pos), mask_penalty, torch.zeros_like(mask_penalty))[pid].unsqueeze(0)
        out_unbatched = m_unbatched(x2d, pid, attn_mask)
        out_batched = m_batched(x3d, pid, attn_mask)
        assert out_batched.shape == (1, 1, 32)
        torch.testing.assert_close(out_unbatched, out_batched.squeeze(0), atol=1e-5, rtol=1e-5)


def test_add_batch_dim_matches_unbatched_under_pruned_rope():
    """The per-kv-group RoPE pruning projection (_project_rope) must still
    differ correctly per layer's own rope_freq_proj when add_batch_dim=True --
    add_batch_dim only affects the shared, pruning-independent base lookup."""
    torch.manual_seed(0)
    g0, g1 = _head_mask([0, 1]), _head_mask([1, 2])
    q_keep = torch.cat([g0, g0, g1, g1])
    k_keep = torch.cat([g0, g1])

    kwargs = dict(kv_num_heads=2, max_cache_len=16, cache_impl="mutable", dtype=torch.float32, attn_impl="manual")
    m_unbatched = MHACausal(32, 4, add_batch_dim=False, **kwargs).eval()
    m_batched = MHACausal(32, 4, add_batch_dim=True, **kwargs).eval()
    m_batched.load_state_dict(m_unbatched.state_dict())  # copy before pruning -- same base weights

    def _hard_prune(mod):
        with torch.no_grad():
            mod.q_proj.weight = torch.nn.Parameter(mod.q_proj.weight[q_keep, :].clone())
            mod.k_proj.weight = torch.nn.Parameter(mod.k_proj.weight[k_keep, :].clone())
        mod.q_proj.out_features = int(q_keep.sum())
        mod.k_proj.out_features = int(k_keep.sum())
        mod.q_proj.register_buffer("rope_prune_mask", q_keep)
        mod.k_proj.register_buffer("rope_prune_mask", k_keep)
        mod.finalize_rope_pruning()

    _hard_prune(m_unbatched)
    _hard_prune(m_batched)
    assert torch.equal(m_unbatched.rope_freq_proj, m_batched.rope_freq_proj)

    inv = precompute_rope_inv_freqs(head_dim=HEAD_DIM, base=1e4)
    cos, sin = precompute_rope_cos_sin_half(inv, torch.tensor(16), torch.tensor(0))
    q_pos = k_pos = torch.arange(16)
    mask_penalty = torch.full((16, 16), float("-inf"))

    for pos in range(3):
        pid = torch.tensor([pos])
        attn_mask = torch.where((q_pos < k_pos), mask_penalty, torch.zeros_like(mask_penalty))[pid].unsqueeze(0)
        rope_pos = (cos[pid], sin[pid])
        x2d = torch.randn(1, 32)
        x3d = x2d.unsqueeze(0)
        out_u = m_unbatched(x2d, pid, attn_mask, rope=rope_pos)
        out_b = m_batched(x3d, pid, attn_mask, rope=rope_pos)
        torch.testing.assert_close(out_u, out_b.squeeze(0), atol=1e-5, rtol=1e-5)


_CONVERTED_INDEXING = {
    "num_blocks_attr": "num_hidden_layers",
    "path_template": "blocks.{block_index}.{path}",
    "path_list": {
        "norm_in":  [],
        "attn":     ["q_proj", "k_proj", "v_proj", "out_proj"],
        "norm_out": [],
        "mlp":      ["gate_proj", "up_proj", "down_proj"],
    },
}


def _tiny_decoder_config(add_batch_dim):
    return CustomDecoderConfigCompress(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_act="silu",
        attn_type="mha_causal",
        mlp_type="mlp_gated",
        norm_type="rmsnorm",
        num_key_value_heads=2,
        max_cache_len=8,
        cache_impl="mutable",
        add_batch_dim=add_batch_dim,
        indexing=_CONVERTED_INDEXING,
    )


def test_transformer_decoder_add_batch_dim_output_shape_and_parity():
    torch.manual_seed(0)
    cfg_unbatched = _tiny_decoder_config(add_batch_dim=False)
    cfg_batched = _tiny_decoder_config(add_batch_dim=True)

    dec_unbatched = TransformerDecoder(config=cfg_unbatched).eval()
    dec_batched = TransformerDecoder(config=cfg_batched).eval()
    dec_batched.load_state_dict(dec_unbatched.state_dict())

    for pos in range(3):
        pid = torch.tensor([pos])
        x2d = torch.randn(1, 16)
        x3d = x2d.unsqueeze(0)
        out_u = dec_unbatched(x2d, pos_id=pid)
        out_b = dec_batched(x3d, pos_id=pid)
        assert out_u.shape == (1, 16)
        assert out_b.shape == (1, 1, 16)
        torch.testing.assert_close(out_u, out_b.squeeze(0), atol=1e-5, rtol=1e-5)
