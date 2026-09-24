"""cache_impl="io_inplace" (MHACausal / TransformerDecoder): the BHSD
tsurgeon::kv_cache_update path must match io_scatter token by token, and a
multi-token call (prompt prefill) must match feeding the same tokens one at a
time. See docs/investigations for why this mode exists (ONNX TensorScatter,
executed in place by TensorRT).
"""
from __future__ import annotations

import pytest
import torch

from transformersurgeon.blocks.config import CustomDecoderConfigCompress
from transformersurgeon.blocks.decoder import TransformerDecoder
from transformersurgeon.blocks.kv_cache_ops import kv_cache_update

MAX_LEN = 16

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


def _decoder(cache_impl, attn_impl="manual"):
    cfg = CustomDecoderConfigCompress(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=48,
        hidden_act="silu",
        attn_type="mha_causal",
        mlp_type="mlp_gated",
        norm_type="rmsnorm",
        num_key_value_heads=2,
        max_cache_len=MAX_LEN,
        cache_impl=cache_impl,
        attn_impl=attn_impl,
        indexing=_CONVERTED_INDEXING,
    )
    return TransformerDecoder(config=cfg).eval()


def _zero_caches(decoder):
    keys = [torch.zeros(b.attn.cache_shape("key")) for b in decoder.blocks]
    values = [torch.zeros(b.attn.cache_shape("value")) for b in decoder.blocks]
    return keys, values


def test_kv_cache_update_is_functional_and_writes_rows():
    cache = torch.zeros(1, 2, MAX_LEN, 4)
    update = torch.randn(1, 2, 3, 4)
    out = kv_cache_update(cache, update, torch.tensor([5]))
    assert torch.count_nonzero(cache) == 0  # input untouched
    torch.testing.assert_close(out[:, :, 5:8], update)
    assert torch.count_nonzero(out[:, :, :5]) == 0 and torch.count_nonzero(out[:, :, 8:]) == 0


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_io_inplace_matches_io_scatter_token_by_token(attn_impl):
    torch.manual_seed(0)
    ref = _decoder("io_scatter", attn_impl)
    dec = _decoder("io_inplace", attn_impl)
    dec.load_state_dict(ref.state_dict())
    rk, rv = _zero_caches(ref)
    k, v = _zero_caches(dec)
    for pos in range(6):
        x = torch.randn(1, 32)
        pid = torch.tensor([pos])
        out_ref, rk, rv = ref(x, pos_id=pid, key_caches=rk, value_caches=rv)
        out, k, v = dec(x, pos_id=pid, key_caches=k, value_caches=v)
        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)
    # Same cache contents, different layout: (L, H, D) vs (1, H, L, D).
    torch.testing.assert_close(k[0][0].transpose(0, 1), rk[0], atol=1e-6, rtol=1e-6)


def test_io_inplace_multi_token_prefill_matches_single_token_steps():
    torch.manual_seed(0)
    dec = _decoder("io_inplace")
    prompt = torch.randn(5, 32)
    start = 3  # prefill into a non-empty cache (e.g. a follow-up turn)

    k, v = _zero_caches(dec)
    for pos in range(start):
        _, k, v = dec(torch.randn(1, 32), pos_id=torch.tensor([pos]), key_caches=k, value_caches=v)
    k_step, v_step = list(k), list(v)
    for i in range(prompt.shape[0]):
        out_step, k_step, v_step = dec(prompt[i : i + 1], pos_id=torch.tensor([start + i]),
                                       key_caches=k_step, value_caches=v_step)

    out_all, k_all, v_all = dec(prompt, pos_id=torch.tensor([start]), key_caches=k, value_caches=v)
    assert out_all.shape == (5, 32)
    torch.testing.assert_close(out_all[-1:], out_step, atol=1e-5, rtol=1e-5)
    for a, b in zip(k_all + v_all, k_step + v_step):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
