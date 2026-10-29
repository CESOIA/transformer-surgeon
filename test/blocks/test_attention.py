"""blocks/attention: kernels, KV-cache atoms and the attention modules composing them."""
import importlib.util

import pytest
import torch

from transformersurgeon.blocks import ATTENTION_KERNELS, KV_CACHES, MHACausal, MHAEncoder
from transformersurgeon.blocks.attention.kernels import attention
from transformersurgeon.blocks.attention.kv_cache import kv_cache_update
from transformersurgeon.blocks.positions.rope import precompute_rope_cos_sin_half, precompute_rope_inv_freqs

L = 8  # cache length


def _naive_attention(q, k, v, mask=None, scale=None):
    """(seq, heads, dim) reference with explicit repeat of the key/value heads."""
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * (scale or q.shape[-1] ** -0.5)
    if mask is not None:
        scores = scores + mask
    return torch.einsum("hqk,khd->hqd", scores.softmax(-1), v)


def test_manual_attention_matches_naive_grouped_attention():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 4, 8), torch.randn(5, 2, 8), torch.randn(5, 2, 8)
    mask = torch.zeros(1, 5)
    mask[0, 4] = -65504.0
    assert torch.allclose(attention(q, k, v, mask), _naive_attention(q, k, v, mask), atol=1e-5)
    # head-major layout and masked softmax give the same result
    hsd = attention(q, k.transpose(0, 1), v.transpose(0, 1), mask, kv_layout="hsd")
    assert torch.allclose(hsd, _naive_attention(q, k, v, mask), atol=1e-5)
    masked = attention(q, k, v, mask, masked_softmax=True)
    assert torch.allclose(masked, _naive_attention(q, k, v, mask), atol=1e-5)


def test_kernel_registry_describes_requirements():
    assert set(ATTENTION_KERNELS) == {"manual", "sdpa", "custom_sdpa", "paged"}
    assert ATTENTION_KERNELS["custom_sdpa"].requires_cache == "mutable"
    assert ATTENTION_KERNELS["paged"].requires_cache == "io_paged"
    assert ATTENTION_KERNELS["paged"].fuses_projection
    assert ATTENTION_KERNELS["manual"].supports_masked_softmax


def test_kv_cache_registry_and_shapes():
    assert set(KV_CACHES) == {"mutable", "io_scatter", "io_concat", "io_inplace", "io_paged"}
    attn = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, cache_impl="io_inplace")
    assert attn.cache_shape("key") == (1, 2, L, 8)
    assert MHACausal(32, 4, kv_num_heads=2, max_cache_len=L).cache_shape("value") == (L, 2, 8)
    assert not KV_CACHES["mutable"].graph_io and KV_CACHES["io_inplace"].multi_token


def test_kv_cache_update_op_is_functional():
    cache = torch.zeros(1, 2, L, 4)
    update = torch.randn(1, 2, 3, 4)
    out = kv_cache_update(cache, update, torch.tensor([2]))
    assert torch.equal(cache, torch.zeros_like(cache))
    assert torch.equal(out[:, :, 2:5], update) and torch.equal(out[:, :, :2], torch.zeros(1, 2, 2, 4))


@pytest.mark.parametrize("cache_impl", ["io_scatter", "io_concat", "io_inplace"])
def test_cache_writes_agree_with_mutable_cache(cache_impl):
    torch.manual_seed(0)
    k, v = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    mutable = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L)
    other = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, cache_impl=cache_impl)
    idx = torch.tensor([3])
    kc_m, _ = mutable.kv_cache.write(mutable, k, v, idx, None, None)
    kc_o, vc_o = other.kv_cache.write(other, k, v, idx, other.key_cache, other.value_cache)
    read = other.kv_cache.read(kc_o)
    read = read.transpose(0, 1) if other.kv_cache.read_layout == "hsd" else read
    assert torch.equal(read, kc_m)


def _decode(attn, steps, rope_table, cache_impl):
    torch.manual_seed(1)
    xs = torch.randn(steps, 32)
    kc = vc = None
    outs = []
    for t in range(steps):
        pos = torch.tensor([t])
        mask = torch.where(torch.arange(L)[None, :] > t, -65504.0, 0.0)
        if cache_impl == "io_inplace":
            mask = mask[None]
        rope = (rope_table[0][pos], rope_table[1][pos])
        out = attn(xs[t:t + 1], pos, mask, key_cache=kc, value_cache=vc, rope=rope)
        if isinstance(out, tuple):
            out, kc, vc = out
        outs.append(out)
    return torch.cat(outs)


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
@pytest.mark.parametrize("cache_impl", ["io_scatter", "io_concat", "io_inplace"])
def test_mha_causal_cache_and_kernel_choices_are_equivalent(cache_impl, attn_impl):
    rope = precompute_rope_cos_sin_half(precompute_rope_inv_freqs(8), L, start_pos=0)
    ref = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L).eval()
    other = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, cache_impl=cache_impl, attn_impl=attn_impl).eval()
    other.load_state_dict(ref.state_dict())
    with torch.no_grad():
        assert torch.allclose(_decode(ref, 5, rope, "mutable"), _decode(other, 5, rope, cache_impl), atol=1e-5)


def test_mha_causal_add_batch_dim_matches_unbatched():
    ref = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L).eval()
    batched = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, add_batch_dim=True).eval()
    batched.load_state_dict(ref.state_dict())
    x, pos, mask = torch.randn(1, 32), torch.tensor([0]), torch.zeros(1, L)
    with torch.no_grad():
        assert torch.allclose(batched(x[None], pos, mask)[0], ref(x, pos, mask), atol=1e-6)


@pytest.mark.parametrize("kwargs, message", [
    ({"cache_impl": "bogus"}, "cache_impl"),
    ({"attn_impl": "bogus"}, "attn_impl"),
    ({"attn_impl": "paged"}, "go together"),
    ({"cache_impl": "io_paged", "attn_impl": "manual"}, "go together"),
    ({"masked_softmax": True, "attn_impl": "sdpa"}, "masked_softmax"),
    ({"attn_impl": "custom_sdpa", "cache_impl": "io_scatter"}, "requires cache_impl='mutable'"),
])
def test_mha_causal_rejects_incompatible_atoms(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, **kwargs)


def test_cache_geometry_follows_pruned_value_heads():
    specs = {"v_proj": {"out_features": 12}, "out_proj": {"in_features": 24}}
    built = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, linear_specs=specs)
    assert built.value_cache.shape == (L, 2, 6)  # sized from the specs at construction
    assert built(torch.randn(1, 32), torch.tensor([0]), torch.zeros(1, L)).shape == (1, 32)

    attn = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L)
    attn.v_proj, attn.out_proj = built.v_proj, built.out_proj  # pruned after construction
    with pytest.raises(RuntimeError, match="finalize_cache_geometry"):
        attn(torch.randn(1, 32), torch.tensor([0]), torch.zeros(1, L))
    attn.finalize_cache_geometry()
    assert attn.value_cache.shape == (L, 2, 6)


def test_paged_cache_matches_inplace_cache():
    from transformersurgeon.blocks.attention.kernels.paged import identity_page_table, num_kv_pages
    torch.manual_seed(0)
    rope = precompute_rope_cos_sin_half(precompute_rope_inv_freqs(8), L, start_pos=0)
    ref = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, cache_impl="io_inplace").eval()
    paged = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, cache_impl="io_paged").eval()
    paged.load_state_dict(ref.state_dict())
    pool = torch.zeros(paged.cache_shape("kv"))
    table = torch.cat([rope[0], rope[1]], -1).squeeze(1).unsqueeze(0).float()
    pages = identity_page_table(num_kv_pages(L))
    x = torch.randn(3, 32)
    kc = vc = None
    with torch.no_grad():
        for t in range(3):
            pos = torch.tensor([t])
            mask = torch.where(torch.arange(L)[None, :] > t, -65504.0, 0.0)[None]
            expected, kc, vc = ref(x[t:t + 1], pos, mask, key_cache=kc, value_cache=vc,
                                   rope=(rope[0][pos], rope[1][pos]))
            start = pos.to(torch.int32)
            out, pool, _ = paged(x[t:t + 1], pos, None, key_cache=pool, paged=(start + 1, table, start, pages))
            assert torch.allclose(out, expected, atol=1e-4)


@pytest.mark.skipif(importlib.util.find_spec("executorch") is None, reason="needs ExecuTorch custom ops")
def test_custom_sdpa_kernel_matches_manual():
    try:
        ref = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, dtype=torch.float32).eval()
        custom = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, dtype=torch.float32,
                           attn_impl="custom_sdpa").eval()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    custom.load_state_dict(ref.state_dict(), strict=False)
    rope = precompute_rope_cos_sin_half(precompute_rope_inv_freqs(8), L, start_pos=0)
    with torch.no_grad():
        assert torch.allclose(_decode(ref, 4, rope, "mutable"), _decode(custom, 4, rope, "mutable"), atol=1e-4)


def test_mha_encoder_is_bidirectional_attention():
    torch.manual_seed(0)
    attn = MHAEncoder(32, 4, kv_num_heads=2).eval()
    x = torch.randn(5, 32)
    q, k, v = attn._project_qkv(x, 5)
    expected = attn.out_proj(_naive_attention(q, k, v).transpose(0, 1).reshape(5, 32))
    assert torch.allclose(attn(x), expected, atol=1e-5)
    sdpa = MHAEncoder(32, 4, kv_num_heads=2, attn_impl="sdpa").eval()
    sdpa.load_state_dict(attn.state_dict())
    assert torch.allclose(sdpa(x), attn(x), atol=1e-5)
    with pytest.raises(ValueError, match="not available for encoder"):
        MHAEncoder(32, 4, attn_impl="paged")
