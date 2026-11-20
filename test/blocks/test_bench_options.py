"""Atoms and options ported from the compression-benchmark branch (bench-d78-fp16, frozen as bench-frozen-2026-11):
the RMSNorm variants, key centering, and the latency-only uniform RoPE of the paged kernel."""
import warnings

import pytest
import torch

from transformersurgeon.blocks import build_block
from transformersurgeon.blocks.norm import (RMSNorm, RMSNormPrescaled, RMSNormPrescaledExactEps, RMSNormThreshold,
                                            RMSNormThresholdSafeSum)


def _rows(scale):
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float64) * scale
    x[0, 3] = 900.0 * scale   # one large row
    return x


@pytest.mark.parametrize("scale", [1e-3, 1.0, 50.0])
def test_threshold_and_exact_eps_norms_equal_the_unscaled_norm(scale):
    x = _rows(scale)
    plain = RMSNorm(64, dtype=torch.float64, eps=1e-6)(x)
    for atom in (RMSNormThreshold(64, dtype=torch.float64, eps=1e-6),
                 RMSNormThresholdSafeSum(64, dtype=torch.float64, eps=1e-6),
                 RMSNormPrescaledExactEps(64, dtype=torch.float64, eps=1e-6)):
        torch.testing.assert_close(atom(x), plain, rtol=1e-6, atol=1e-9)


def test_threshold_leaves_small_rows_untouched_and_bounds_large_ones():
    norm = RMSNormThreshold(64, threshold=128.0)
    x = torch.full((2, 64), 3.0)
    x[1, 0] = 1000.0
    assert torch.equal(norm._rescale(x)[0], x[0])
    assert norm._rescale(x)[1].abs().max().item() == pytest.approx(128.0)
    # the max prescale adds eps to the RESCALED variance: for a row with max|x| = 1000 it is not the unscaled norm
    big = torch.full((1, 64), 1.0, dtype=torch.float64)
    big[0, 0] = 1000.0
    plain = RMSNorm(64, dtype=torch.float64, eps=1e-1)(big)
    assert not torch.allclose(RMSNormPrescaled(64, dtype=torch.float64, eps=1e-1)(big), plain, rtol=1e-6)
    torch.testing.assert_close(RMSNormPrescaledExactEps(64, dtype=torch.float64, eps=1e-1)(big), plain)


def test_threshold_comes_from_the_stack_configuration():
    class Config:
        rmsnorm_threshold = 16.0
    assert build_block("norm", "rmsnorm_threshold", 8, config=Config()).threshold == 16.0
    assert build_block("norm", "rmsnorm_threshold", 8).threshold == 128.0
    assert build_block("norm", "rmsnorm_threshold", 8, config=Config(), threshold=4.0).threshold == 4.0



def _attention(cache_impl="mutable", attn_impl="manual", center_keys=False):
    from transformersurgeon.blocks.attention.causal import MHACausal
    torch.manual_seed(0)
    attn = MHACausal(32, 4, kv_num_heads=2, attn_impl=attn_impl, cache_impl=cache_impl, max_cache_len=8,
                     dtype=torch.float32, center_keys=center_keys)
    return attn


@pytest.mark.parametrize("cache_impl", ["mutable", "io_scatter", "io_inplace"])
def test_center_keys_leaves_attention_unchanged(cache_impl):
    ref, cen = _attention(cache_impl), _attention(cache_impl, center_keys=True)
    cen.load_state_dict(ref.state_dict())
    torch.manual_seed(1)
    steps = [torch.randn(1, 32) for _ in range(5)]
    caches = {}
    for attn in (ref, cen):
        kc = vc = None
        if attn.kv_cache.graph_io:
            kc, vc = attn.zero_cache("key"), attn.zero_cache("value")
        outs = []
        for pos, x in enumerate(steps):
            mask = torch.where(torch.arange(8)[None, :] > pos, -1e4, 0.0)
            if attn.kv_cache.read_layout == "hsd":
                mask = mask.unsqueeze(0)
            cos, sin = torch.ones(1, 1, 4), torch.zeros(1, 1, 4)
            out = attn(x, torch.tensor([pos]), mask, key_cache=kc, value_cache=vc, rope=(cos, sin))
            if attn.kv_cache.graph_io:
                out, kc, vc = out
            outs.append(out)
        caches[id(attn)] = torch.stack(outs)
    torch.testing.assert_close(caches[id(cen)], caches[id(ref)], rtol=1e-5, atol=1e-5)


def test_center_keys_is_refused_with_a_kernel_that_keeps_its_own_cache():
    with pytest.raises(ValueError, match="center_keys=True needs a kernel"):
        _attention("io_paged", attn_impl="paged", center_keys=True)


def test_uniform_rope_option_is_declared_for_the_paged_backends_only():
    from transformersurgeon.export.operations import check_choice, options_for
    assert True in options_for("tensorrt", "paged_rope") and True in options_for("onnx", "paged_rope")
    for backend in ("xnnpack", "qnn"):
        with pytest.raises(ValueError, match="not supported"):
            check_choice(backend, "paged_rope", True)
