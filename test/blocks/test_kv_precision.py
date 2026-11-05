"""blocks/attention/kv_precision: integer KV-cache rows, and the causal attention that stores them."""
import pytest
import torch

from transformersurgeon.blocks import BLOCK_CATALOG, KV_PRECISIONS, MHACausal
from transformersurgeon.blocks.attention.kv_precision import SCALE_BYTES
from transformersurgeon.blocks.positions.rope import precompute_rope_cos_sin_half, precompute_rope_inv_freqs

L = 8  # cache length
INTEGER = ["int8", "int4"]
CACHES = ["mutable", "io_scatter", "io_concat", "io_inplace"]
# Worst error of a decoded element relative to its row's max |x|: half a code step, with the
# scale rounded up by at most 1/256.
ROW_ERROR = {"int8": 0.5 / 127 * (1 + 1 / 256), "int4": 0.5 / 7 * (1 + 1 / 256)}
ROPE = precompute_rope_cos_sin_half(precompute_rope_inv_freqs(8), L, start_pos=0)


def _rows(seed=0):
    """Rows of very different magnitudes (tiny values next to keys with large biases)."""
    torch.manual_seed(seed)
    return torch.randn(6, 3, 16) * torch.tensor([1e-3, 1.0, 300.0]).view(1, 3, 1)


def _attn(cache_impl="mutable", kv_precision="full", **kwargs):
    return MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, cache_impl=cache_impl, kv_precision=kv_precision,
                     **kwargs).eval()


def _mask(positions, cache_impl):
    mask = torch.where(torch.arange(L)[None, :] > positions[:, None], -65504.0, 0.0)
    return mask[None] if cache_impl == "io_inplace" else mask


def _decode(attn, xs, cache_impl):
    """Feed ``xs`` one token per step; returns the outputs and the last key cache (graph-I/O caches)."""
    kc = vc = None
    outs = []
    for t in range(xs.shape[0]):
        pos = torch.tensor([t])
        out = attn(xs[t:t + 1], pos, _mask(pos, cache_impl), key_cache=kc, value_cache=vc,
                   rope=(ROPE[0][pos], ROPE[1][pos]))
        if isinstance(out, tuple):
            out, kc, vc = out
        outs.append(out)
    return torch.cat(outs), kc


# --------------------------------------------------------------------------- #
# The precision atoms
# --------------------------------------------------------------------------- #
def test_registry_and_catalog_list_the_same_precisions():
    assert set(KV_PRECISIONS) == set(BLOCK_CATALOG["kv_precision"]) == {"full", "int8", "int4"}


@pytest.mark.parametrize("name, width, dtype", [
    ("full", 16, torch.float16), ("int8", 16 + SCALE_BYTES, torch.uint8), ("int4", 8 + SCALE_BYTES, torch.uint8),
])
def test_storage_dtype_and_row_width(name, width, dtype):
    precision = KV_PRECISIONS[name]
    assert precision.row_width(16) == width and precision.storage_dtype(torch.float16) == dtype


def test_full_precision_adds_nothing():
    x = _rows()
    assert KV_PRECISIONS["full"].encode(x) is x and KV_PRECISIONS["full"].decode(x, torch.float16) is x


@pytest.mark.parametrize("name", INTEGER)
def test_integer_rows_round_trip_within_half_a_code_step(name):
    precision = KV_PRECISIONS[name]
    x = _rows()
    stored = precision.encode(x)
    assert stored.dtype == torch.uint8 and stored.shape == (6, 3, precision.row_width(16))
    error = (precision.decode(stored, torch.float32) - x).abs()
    assert (error <= ROW_ERROR[name] * x.abs().amax(-1, keepdim=True)).all()
    # The scale is rounded up, so codes never reach the clamp: offset codes stay >= 1.
    codes = stored[..., :-SCALE_BYTES]
    if name == "int8":
        assert codes.min() >= 1
    else:
        assert (codes % 16).min() >= 1 and (codes // 16).min() >= 1


@pytest.mark.parametrize("name", INTEGER)
def test_rows_are_encoded_independently(name):
    """Prefill (many rows per call) and decode (one row) must store the same bytes."""
    precision = KV_PRECISIONS[name]
    x = _rows()
    assert torch.equal(precision.encode(x)[2:3], precision.encode(x[2:3]))


@pytest.mark.parametrize("name", INTEGER)
def test_fresh_cache_and_zero_rows_decode_to_zero(name):
    precision = KV_PRECISIONS[name]
    fresh = torch.zeros(4, 2, precision.row_width(16), dtype=torch.uint8)
    assert precision.decode(fresh, torch.float32).abs().max() < 1e-30
    assert precision.decode(fresh, torch.float16).abs().max() == 0
    zeros = torch.zeros(2, 16)
    assert torch.equal(precision.decode(precision.encode(zeros), torch.float32), zeros)


def test_int4_needs_an_even_head_dim():
    with pytest.raises(ValueError, match="even head dim"):
        KV_PRECISIONS["int4"].row_width(15)
    with pytest.raises(ValueError, match="even head dim"):
        MHACausal(36, 4, kv_num_heads=2, max_cache_len=L, kv_precision="int4")  # head dim 9


# --------------------------------------------------------------------------- #
# Causal attention with an integer cache
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cache_impl", CACHES)
@pytest.mark.parametrize("name, tolerance", [("int8", 0.02), ("int4", 0.2)])
def test_integer_cache_stays_close_to_the_float_cache(cache_impl, name, tolerance):
    ref, quant = _attn(), _attn(cache_impl, name)
    quant.load_state_dict(ref.state_dict())  # the caches are not persistent: weights only
    assert quant.cache_dtype == torch.uint8
    assert quant.cache_shape("key")[-1] == KV_PRECISIONS[name].row_width(8)
    xs = torch.randn(5, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected, _ = _decode(ref, xs, "mutable")
        got, _ = _decode(quant, xs, cache_impl)
    assert (got - expected).norm() / expected.norm() < tolerance


@pytest.mark.parametrize("name", INTEGER)
@pytest.mark.parametrize("cache_impl", ["io_scatter", "io_concat", "io_inplace"])
def test_every_cache_atom_stores_the_same_integer_rows(cache_impl, name):
    mutable, other = _attn("mutable", name), _attn(cache_impl, name)
    other.load_state_dict(mutable.state_dict())
    xs = torch.randn(4, 32, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        out_mutable, _ = _decode(mutable, xs, "mutable")
        out_other, key_cache = _decode(other, xs, cache_impl)
    stored = other.kv_cache.read(key_cache)
    stored = stored.transpose(0, 1) if other.kv_cache.read_layout == "hsd" else stored
    assert torch.equal(stored, mutable.key_cache)
    assert torch.allclose(out_other, out_mutable, atol=1e-5)


@pytest.mark.parametrize("name", INTEGER)
def test_prefill_stores_the_rows_decode_stores(name):
    step, prefill = _attn("io_inplace", name), _attn("io_inplace", name)
    prefill.load_state_dict(step.state_dict())
    xs = torch.randn(3, 32, generator=torch.Generator().manual_seed(3))
    positions = torch.arange(3)
    with torch.no_grad():
        out_steps, key_steps = _decode(step, xs, "io_inplace")
        out_prefill, key_prefill, _ = prefill(xs, torch.tensor([0]), _mask(positions, "io_inplace"),
                                              rope=(ROPE[0][positions], ROPE[1][positions]))
    assert torch.equal(key_prefill, key_steps)
    assert torch.allclose(out_prefill, out_steps, atol=1e-5)


@pytest.mark.parametrize("kwargs, message", [
    ({"kv_precision": "int2"}, "kv_precision"),
    ({"kv_precision": "int8", "cache_impl": "io_paged"}, "own float cache"),
    ({"kv_precision": "int8", "attn_impl": "custom_sdpa"}, "own float cache"),
])
def test_integer_precision_rejects_kernels_with_their_own_cache(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, **kwargs)


def test_integer_cache_geometry_follows_pruned_value_heads():
    specs = {"v_proj": {"out_features": 12}, "out_proj": {"in_features": 24}}
    built = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, linear_specs=specs, kv_precision="int8")
    assert built.value_cache.shape == (L, 2, 6 + SCALE_BYTES) and built.value_cache.dtype == torch.uint8

    attn = MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, kv_precision="int8")
    attn.v_proj, attn.out_proj = built.v_proj, built.out_proj  # pruned after construction
    attn.finalize_cache_geometry()
    assert attn.value_cache.shape == (L, 2, 6 + SCALE_BYTES) and attn.value_cache.dtype == torch.uint8


class _Step(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, x, pos, mask, cos, sin, *caches):
        return self.attn(x, pos, mask, *caches, rope=(cos, sin))


@pytest.mark.parametrize("cache_impl, name", [("mutable", "int8"), ("io_inplace", "int4"), ("io_scatter", "int8")])
def test_integer_cache_exports_and_matches_eager(cache_impl, name):
    """torch.export traces the encode/decode (no data-dependent control flow) and the exported
    program, run step by step, gives the eager outputs."""
    traced, eager = _attn(cache_impl, name), _attn(cache_impl, name)
    eager.load_state_dict(traced.state_dict())
    graph_io = cache_impl != "mutable"
    caches = (traced.key_cache.clone(), traced.value_cache.clone()) if graph_io else ()
    pos = torch.tensor([0])
    example = (torch.randn(1, 32), pos, _mask(pos, cache_impl), ROPE[0][pos], ROPE[1][pos], *caches)
    program = torch.export.export(_Step(traced), example).module()

    xs = torch.randn(4, 32, generator=torch.Generator().manual_seed(4))
    eager_caches, program_caches = caches, caches
    with torch.no_grad():
        for t in range(4):
            pos = torch.tensor([t])
            step = (xs[t:t + 1], pos, _mask(pos, cache_impl), ROPE[0][pos], ROPE[1][pos])
            expected = _Step(eager)(*step, *eager_caches)
            got = program(*step, *program_caches)
            if graph_io:
                expected, *eager_caches = expected
                got, *program_caches = got
                assert program_caches[0].dtype == torch.uint8
            assert torch.allclose(got, expected, atol=1e-6)
