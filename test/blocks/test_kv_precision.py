"""blocks/attention/kv_precision: integer, FP8 and NVFP4 KV-cache rows, and the causal attention that stores them."""
import pytest
import torch

from _helpers import capabilities as caps
from transformersurgeon.blocks import BLOCK_CATALOG, KV_PRECISIONS, MHACausal
from transformersurgeon.blocks.attention.kv_precision import FP8_MAX, NVFP4_BLOCK, SCALE_BYTES
from transformersurgeon.blocks.attention.kv_precision.nvfp4 import E2M1_VALUES
from transformersurgeon.blocks.positions.rope import precompute_rope_cos_sin_half, precompute_rope_inv_freqs

L = 8  # cache length
INTEGER = ["int8", "int4"]
CACHES = ["mutable", "io_scatter", "io_concat", "io_inplace", "io_rows"]
BHSD = ("io_inplace", "io_rows")
# Worst error of a decoded element relative to its row's max |x|: half a code step, with the
# scale rounded up by at most 1/256.
ROW_ERROR = {"int8": 0.5 / 127 * (1 + 1 / 256), "int4": 0.5 / 7 * (1 + 1 / 256)}
# RoPE tables by head dim: 8 for the small attention, 16 for NVFP4 (one block of 16 per row).
ROPES = {d: precompute_rope_cos_sin_half(precompute_rope_inv_freqs(d), L, start_pos=0) for d in (8, 16)}
ROPE = ROPES[8]


def _rows(seed=0):
    """Rows of very different magnitudes (tiny values next to keys with large biases)."""
    torch.manual_seed(seed)
    return torch.randn(6, 3, 16) * torch.tensor([1e-3, 1.0, 300.0]).view(1, 3, 1)


def _attn(cache_impl="mutable", kv_precision="full", embed=32, **kwargs):
    """4 heads (2 KV heads) of dim ``embed / 4``: 8 by default, 16 with ``embed=64`` (NVFP4)."""
    return MHACausal(embed, 4, kv_num_heads=2, max_cache_len=L, cache_impl=cache_impl, kv_precision=kv_precision,
                     **kwargs).eval()


def _mask(positions, cache_impl):
    mask = torch.where(torch.arange(L)[None, :] > positions[:, None], -65504.0, 0.0)
    return mask[None] if cache_impl in BHSD else mask


def _write_rows(cache, rows, pos):
    """The caller's side of io_rows: write the returned rows at ``pos`` (one-byte rows as bytes;
    a ``(rows, scales)`` cache part by part)."""
    if isinstance(cache, tuple):
        for part, part_rows in zip(cache, rows):
            _write_rows(part, part_rows, pos)
        return
    positions = pos + torch.arange(rows.shape[2])
    if cache.element_size() == 1:
        cache, rows = cache.view(torch.uint8), rows.view(torch.uint8)
    cache.index_copy_(2, positions, rows)


def _decode(attn, xs, cache_impl):
    """Feed ``xs`` one token per step; returns the outputs and the last key cache (graph-I/O caches;
    for io_rows the caller's cache the returned rows were written into)."""
    kc = vc = None
    if cache_impl == "io_rows":
        kc, vc = attn.zero_cache("key"), attn.zero_cache("value")
    cos, sin = ROPES[attn.key_head_dim]
    outs = []
    for t in range(xs.shape[0]):
        pos = torch.tensor([t])
        out = attn(xs[t:t + 1], pos, _mask(pos, cache_impl), key_cache=kc, value_cache=vc,
                   rope=(cos[pos], sin[pos]))
        if isinstance(out, tuple):
            out, k_out, v_out = out
            if cache_impl == "io_rows":
                _write_rows(kc, k_out, pos)
                _write_rows(vc, v_out, pos)
            else:
                kc, vc = k_out, v_out
        outs.append(out)
    return torch.cat(outs), kc


# --------------------------------------------------------------------------- #
# The precision atoms
# --------------------------------------------------------------------------- #
def test_registry_and_catalog_list_the_same_precisions():
    assert set(KV_PRECISIONS) == set(BLOCK_CATALOG["kv_precision"]) == {"full", "int8", "int4", "fp8", "nvfp4"}
    assert {name: p.requires_cache for name, p in KV_PRECISIONS.items()} == {
        "full": None, "int8": None, "int4": None, "fp8": "io_rows", "nvfp4": "io_rows"}
    assert {name for name, p in KV_PRECISIONS.items() if p.scale_dtype is not None} == {"nvfp4"}


@pytest.mark.parametrize("name, width, dtype", [
    ("full", 16, torch.float16), ("int8", 16 + SCALE_BYTES, torch.uint8), ("int4", 8 + SCALE_BYTES, torch.uint8),
    ("fp8", 16, torch.float8_e4m3fn),
    pytest.param("nvfp4", 8, getattr(torch, "float4_e2m1fn_x2", None), marks=caps.requires_float4),
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


def test_fp8_rows_round_to_the_nearest_e4m3_and_saturate():
    precision = KV_PRECISIONS["fp8"]
    x = _rows()
    stored = precision.encode(x)
    assert stored.dtype == torch.float8_e4m3fn and stored.shape == x.shape
    decoded = precision.decode(stored, torch.float32)
    # 3 mantissa bits: at most half a step, 2**-4 of the value, in the normal range [2**-6, 448];
    # below it (subnormals) the step is fixed at 2**-9; above it, values saturate.
    normal = (x.abs() >= 2.0 ** -6) & (x.abs() <= FP8_MAX)
    assert ((decoded - x).abs()[normal] <= 2.0 ** -4 * x.abs()[normal]).all()
    assert ((decoded - x).abs()[x.abs() < 2.0 ** -6] <= 2.0 ** -10).all()
    assert (decoded[x.abs() > FP8_MAX].abs() == FP8_MAX).all() and (x.abs() > FP8_MAX).any()
    big = precision.decode(precision.encode(torch.tensor([1e4, -1e4, 448.0, 0.0])), torch.float32)
    assert big.tolist() == [FP8_MAX, -FP8_MAX, 448.0, 0.0]
    assert torch.equal(precision.encode(x)[2:3].view(torch.uint8), precision.encode(x[2:3]).view(torch.uint8))
    assert precision.decode(torch.zeros(4, 2, 16, dtype=torch.float8_e4m3fn), torch.float16).abs().max() == 0


@pytest.mark.parametrize("cache_impl", ["mutable", "io_scatter", "io_concat", "io_inplace"])
def test_fp8_needs_the_rows_cache(cache_impl):
    with pytest.raises(ValueError, match="needs cache_impl='io_rows'"):
        MHACausal(32, 4, kv_num_heads=2, max_cache_len=L, cache_impl=cache_impl, kv_precision="fp8")


def _e4m3_byte(value):
    return torch.tensor(value).to(torch.float8_e4m3fn).view(torch.uint8).item()


@caps.requires_float4
def test_nvfp4_rows_hold_e2m1_codes_with_one_e4m3_scale_per_16_elements():
    precision = KV_PRECISIONS["nvfp4"]
    assert (precision.row_width(64), precision.scale_width(64), precision.scale_dtype) == (32, 4, torch.float8_e4m3fn)
    # Scale 0.5 (block max 3): every E2M1 magnitude, both signs, is stored exactly.
    block = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES)) * 0.5
    rows, scales = precision.encode(block)
    assert (rows.dtype, rows.shape, scales.dtype, scales.shape) == \
        (torch.float4_e2m1fn_x2, (8,), torch.float8_e4m3fn, (1,))
    assert scales.view(torch.uint8).tolist() == [_e4m3_byte(0.5)]
    assert rows.view(torch.uint8).tolist() == [code | (code + 1) << 4 for code in range(0, 16, 2)]  # 2i low
    assert torch.equal(precision.decode((rows, scales), torch.float32), block)


@caps.requires_float4
def test_nvfp4_rounds_to_the_nearest_even_code_and_saturates():
    precision = KV_PRECISIONS["nvfp4"]

    def round_trip(block):
        return precision.decode(precision.encode(torch.tensor(block)), torch.float32).tolist()

    # Scale 1 (block max 6): ties between two magnitudes go to the even code.
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.1]
    assert round_trip(ties) == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0, 0.0]
    # Block max 6.375: the scale 1.0625 is an E4M3 tie and rounds down to 1, the max saturates at 6.
    assert round_trip([6.375] + [0.0] * 15)[0] == 6.0
    # Scales saturate at 448; a block below 6 * 2**-10 gets scale 0 and decodes to zero.
    assert round_trip([1e5] + [0.0] * 15)[0] == 6 * FP8_MAX
    assert round_trip([5e-3] * 16) == [0.0] * 16
    zeros = torch.zeros(2, 32)
    assert torch.equal(precision.decode(precision.encode(zeros), torch.float32), zeros)


@caps.requires_float4
def test_nvfp4_error_is_relative_to_each_block():
    """Half the widest E2M1 step (1 for magnitudes in [4, 6]) times the scale, which RNE keeps within
    1/16 of max/6: at most 17/96 of the block's max |x| (normal scales); blocks of very different
    magnitudes in one row do not disturb each other."""
    precision = KV_PRECISIONS["nvfp4"]
    torch.manual_seed(0)
    x = torch.randn(6, 3, 64) * torch.tensor([0.1, 1.0, 30.0, 300.0]).repeat_interleave(NVFP4_BLOCK)
    blocks = x.unflatten(-1, (-1, NVFP4_BLOCK))
    error = (precision.decode(precision.encode(x), torch.float32) - x).unflatten(-1, (-1, NVFP4_BLOCK))
    assert (error.abs() <= 17 / 96 * blocks.abs().amax(-1, keepdim=True)).all()
    rows, scales = precision.encode(x)
    one_row, one_scales = precision.encode(x[2:3])   # prefill and decode store the same bytes
    assert torch.equal(rows[2:3].view(torch.uint8), one_row.view(torch.uint8))
    assert torch.equal(scales[2:3].view(torch.uint8), one_scales.view(torch.uint8))


@caps.requires_float4
def test_nvfp4_keeps_the_sign_of_zero():
    """-0.0, and negatives that round to zero, keep the sign bit (code 8), as TensorRT stores them."""
    rows, _ = KV_PRECISIONS["nvfp4"].encode(torch.tensor([-0.0, 0.0, -0.01, 0.01] + [6.0] * 12))
    assert rows.view(torch.uint8).tolist()[:2] == [0x08, 0x08]


@pytest.mark.parametrize("cache_impl", ["mutable", "io_scatter", "io_concat", "io_inplace"])
def test_nvfp4_needs_the_rows_cache(cache_impl):
    with pytest.raises(ValueError, match="needs cache_impl='io_rows'"):
        _attn(cache_impl, "nvfp4", embed=64)


@caps.requires_float4
def test_nvfp4_needs_a_head_dim_multiple_of_16():
    with pytest.raises(ValueError, match="multiple of 16"):
        KV_PRECISIONS["nvfp4"].row_width(24)
    with pytest.raises(ValueError, match="multiple of 16"):
        _attn("io_rows", "nvfp4")  # head dim 8


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
@pytest.mark.parametrize("cache_impl", ["io_scatter", "io_concat", "io_inplace", "io_rows"])
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


def test_fp8_cache_stays_close_to_the_float_cache():
    ref, fp8 = _attn("io_rows"), _attn("io_rows", "fp8")
    fp8.load_state_dict(ref.state_dict())
    assert fp8.cache_dtype == torch.float8_e4m3fn and fp8.cache_shape("key") == (1, 2, L, 8)
    xs = torch.randn(5, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected, float_keys = _decode(ref, xs, "io_rows")
        got, fp8_keys = _decode(fp8, xs, "io_rows")
    assert (got - expected).norm() / expected.norm() < 0.06
    # The caller's cache holds exactly the FP8 rounding of the float keys.
    assert torch.equal(fp8_keys.view(torch.uint8), KV_PRECISIONS["fp8"].encode(float_keys).view(torch.uint8))


@caps.requires_float4
def test_nvfp4_cache_is_a_pair_of_rows_and_block_scales():
    attn = _attn("io_rows", "nvfp4", embed=64)
    assert (attn.cache_dtype, attn.cache_scale_dtype) == (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)
    assert (attn.cache_shape("key"), attn.cache_shape("key_scale")) == ((1, 2, L, 8), (1, 2, L, 1))
    rows, scales = attn.zero_cache("value")
    assert (rows.shape, rows.dtype, scales.shape, scales.dtype) == \
        ((1, 2, L, 8), torch.float4_e2m1fn_x2, (1, 2, L, 1), torch.float8_e4m3fn)
    assert (attn.key_cache_scale.shape, attn.value_cache_scale.dtype) == ((1, 2, L, 1), torch.float8_e4m3fn)
    assert _attn("io_rows", "fp8").zero_cache("key").dtype == torch.float8_e4m3fn  # one tensor otherwise


@caps.requires_float4
def test_nvfp4_cache_stays_close_to_the_float_cache():
    ref, fp4 = _attn("io_rows", embed=64), _attn("io_rows", "nvfp4", embed=64)
    fp4.load_state_dict(ref.state_dict())
    xs = torch.randn(5, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected, float_keys = _decode(ref, xs, "io_rows")
        got, (fp4_keys, fp4_scales) = _decode(fp4, xs, "io_rows")
    assert (got - expected).norm() / expected.norm() < 0.15
    # The caller's cache holds exactly the NVFP4 encoding of the float keys.
    rows, scales = KV_PRECISIONS["nvfp4"].encode(float_keys)
    assert torch.equal(fp4_keys.view(torch.uint8), rows.view(torch.uint8))
    assert torch.equal(fp4_scales.view(torch.uint8), scales.view(torch.uint8))


@pytest.mark.parametrize("name", ["full", *INTEGER])
def test_rows_cache_returns_the_rows_the_inplace_cache_stores(name):
    """io_rows computes what io_inplace computes; only the cache write moves to the caller."""
    inplace, rows = _attn("io_inplace", name), _attn("io_rows", name)
    rows.load_state_dict(inplace.state_dict())
    xs = torch.randn(4, 32, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        out_inplace, key_inplace = _decode(inplace, xs, "io_inplace")
        out_rows, key_rows = _decode(rows, xs, "io_rows")
    assert torch.equal(key_rows, key_inplace) and torch.equal(out_rows, out_inplace)


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


@pytest.mark.parametrize("name", ["full", "int4", "fp8", pytest.param("nvfp4", marks=caps.requires_float4)])
def test_rows_cache_prefill_returns_the_rows_decode_returns(name):
    embed = 64 if name == "nvfp4" else 32
    step, prefill = _attn("io_rows", name, embed=embed), _attn("io_rows", name, embed=embed)
    prefill.load_state_dict(step.state_dict())
    xs = torch.randn(3, embed, generator=torch.Generator().manual_seed(3))
    positions = torch.arange(3)
    cos, sin = ROPES[step.key_head_dim]
    with torch.no_grad():
        out_steps, key_steps = _decode(step, xs, "io_rows")
        out_prefill, rows, _ = prefill(xs, torch.tensor([0]), _mask(positions, "io_rows"),
                                       key_cache=prefill.zero_cache("key"), value_cache=prefill.zero_cache("value"),
                                       rope=(cos[positions], sin[positions]))
    decode = KV_PRECISIONS[name].decode
    first = rows[0] if name == "nvfp4" else rows
    assert first.shape == (1, 2, 3, prefill.cache_width("key"))
    filled = tuple(c[:, :, :3] for c in key_steps) if name == "nvfp4" else key_steps[:, :, :3]
    assert torch.allclose(decode(rows, torch.float32), decode(filled, torch.float32), atol=1e-6)
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


def _clone(cache):
    return tuple(c.clone() for c in cache) if isinstance(cache, tuple) else cache.clone()


@pytest.mark.parametrize("cache_impl, name", [("mutable", "int8"), ("io_inplace", "int4"), ("io_scatter", "int8"),
                                             ("io_rows", "int8"), ("io_rows", "fp8"),
                                             pytest.param("io_rows", "nvfp4", marks=caps.requires_float4)])
def test_integer_cache_exports_and_matches_eager(cache_impl, name):
    """torch.export traces the encode/decode (no data-dependent control flow) and the exported
    program, run step by step, gives the eager outputs (io_rows: the caller writes the rows)."""
    embed = 64 if name == "nvfp4" else 32
    traced, eager = _attn(cache_impl, name, embed=embed), _attn(cache_impl, name, embed=embed)
    eager.load_state_dict(traced.state_dict())
    graph_io = cache_impl != "mutable"
    caches = (traced.zero_cache("key"), traced.zero_cache("value")) if graph_io else ()
    cos, sin = ROPES[traced.key_head_dim]
    pos = torch.tensor([0])
    example = (torch.randn(1, embed), pos, _mask(pos, cache_impl), cos[pos], sin[pos], *caches)
    program = torch.export.export(_Step(traced), example).module()

    xs = torch.randn(4, embed, generator=torch.Generator().manual_seed(4))
    eager_caches, program_caches = [_clone(c) for c in caches], [_clone(c) for c in caches]
    with torch.no_grad():
        for t in range(4):
            pos = torch.tensor([t])
            step = (xs[t:t + 1], pos, _mask(pos, cache_impl), cos[pos], sin[pos])
            expected = _Step(eager)(*step, *eager_caches)
            got = program(*step, *program_caches)
            if cache_impl == "io_rows":
                expected, *eager_rows = expected
                got, *program_rows = got
                first = program_rows[0][0] if name == "nvfp4" else program_rows[0]
                assert first.dtype == traced.cache_dtype
                for caches, rows in ((eager_caches, eager_rows), (program_caches, program_rows)):
                    for cache, row in zip(caches, rows):
                        _write_rows(cache, row, pos)
            elif graph_io:
                expected, *eager_caches = expected
                got, *program_caches = got
                assert program_caches[0].dtype == torch.uint8
            assert torch.allclose(got, expected, atol=1e-6)
