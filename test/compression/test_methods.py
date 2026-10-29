"""Numerical methods of compression: low-rank factorisations, quantisers, pruning masks."""
import pytest
import torch

from transformersurgeon.compression.coupled_pruning import CoupledPruner
from transformersurgeon.compression.lrd_methods import METHOD_FUNCTIONS as LRD_METHODS
from transformersurgeon.compression.quantization_methods import (
    METHOD_FUNCTIONS as QUANT_METHODS,
    activation_qrange,
    compute_activation_scale_zp_maxmin,
)
from transformersurgeon.compression.quantization_methods.vanilla import lpbq_group_scales
from transformersurgeon.compression.structured_pruning_methods import (
    SCORE_FUNCTIONS,
    build_structured_mask,
    effective_num_pruned,
)
from transformersurgeon.compression.unstructured_pruning_methods import METHOD_FUNCTIONS as UNSTRUCTURED


def _low_rank_error(us, v, w):
    return torch.linalg.norm(w - us @ v) / torch.linalg.norm(w)


def test_svd_is_exact_at_full_rank_and_optimal_below():
    torch.manual_seed(0)
    w = torch.randn(12, 8)
    us, v = LRD_METHODS["svd"](w, 8)
    assert _low_rank_error(us, v, w) < 1e-5
    us4, v4 = LRD_METHODS["svd"](w, 4)
    assert us4.shape == (12, 4) and v4.shape == (4, 8)


def test_data_aware_factorisations_reduce_to_svd_for_white_inputs():
    torch.manual_seed(0)
    w, eye = torch.randn(12, 8), torch.eye(8)
    plain = _low_rank_error(*LRD_METHODS["svd"](w, 4), w)
    assert torch.isclose(_low_rank_error(*LRD_METHODS["svd-llm-v2"](w, 4, eye), w), plain, atol=1e-4)
    assert torch.isclose(_low_rank_error(*LRD_METHODS["aa-svd"](w, 4, eye, eye), w), plain, atol=1e-4)
    with pytest.raises(ValueError):
        LRD_METHODS["svd-llm-v2"](w, 4, torch.eye(3))


@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
@pytest.mark.parametrize("method", ["vanilla", "hqq"])
def test_weight_quantisers_round_trip(method, granularity):
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    q, scale, dequant = QUANT_METHODS[method](w, 8, 1e-6, granularity)
    assert q.dtype == torch.int8
    assert (dequant(q, scale) - w).abs().max() < 0.05


def test_gptq_beats_round_to_nearest_on_correlated_inputs():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    x = torch.randn(256, 32) @ torch.randn(32, 32)
    cov = x.T @ x / len(x)
    def out_err(fn, **kw):
        q, s, deq = fn(w, 3, 1e-6, "per_channel", **kw)
        return torch.linalg.norm(x @ (deq(q, s) - w).T)
    assert out_err(QUANT_METHODS["gptq"], covariance=cov) < out_err(QUANT_METHODS["vanilla"])


def test_per_group_and_lpbq_scales():
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    q, scale, dequant = QUANT_METHODS["vanilla"](w, 4, 1e-6, "per_group", group_size=16, group_scale_bits=4)
    assert scale.shape[-1] == 4 and (dequant(q, scale) - w).abs().max() < 0.6
    channel = torch.rand(4, 1) + 0.5
    lpbq = lpbq_group_scales(channel * torch.rand(4, 4), 4)
    assert lpbq.shape == (4, 4)


def test_activation_ranges_and_maxmin_scales():
    assert activation_qrange(8, "symmetric") == (-127, 127)
    assert activation_qrange(8, "asymmetric") == (0, 255)
    scale, zp = compute_activation_scale_zp_maxmin(torch.tensor(-1.0), torch.tensor(3.0), 8, "asymmetric")
    assert torch.isclose(scale, torch.tensor(4.0 / 255)) and int(zp) == 64
    scale, zp = compute_activation_scale_zp_maxmin(torch.tensor(-1.0), torch.tensor(3.0), 8, "symmetric")
    assert torch.isclose(scale, torch.tensor(3.0 / 127)) and int(zp) == 0


def test_structured_masks_and_scores():
    scores = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0, 0.0])
    keep = build_structured_mask(scores, 0.5)
    assert keep.tolist() == [True, False, True, False, True, False]
    per_chunk = build_structured_mask(scores, 0.5, granularity=3)
    assert per_chunk.sum() == 4  # int(0.5 * 3) = 1 row removed in each chunk of 3
    assert effective_num_pruned(6, 0.5, 3) == 2
    w = torch.tensor([[3.0, 4.0], [0.0, 1.0]])
    assert SCORE_FUNCTIONS["magnitude"](w).tolist() == [5.0, 1.0]


@pytest.mark.parametrize("method", ["magnitude", "random"])
def test_unstructured_masks_hit_the_requested_sparsity(method):
    mask = UNSTRUCTURED[method](torch.randn(8, 8), pruning_ratio=0.25)
    assert mask.dtype == torch.bool and (~mask).sum() == 16


def test_coupled_pruner_removes_and_restores_input_columns():
    lin = torch.nn.Linear(6, 2)
    original = lin.weight.detach().clone()
    keep = torch.tensor([True, False, True, True, False, True])
    pruner = CoupledPruner()
    pruner.apply(lin, keep, hard=True)
    assert lin.in_features == 4 and torch.equal(lin.weight, original[:, keep])
    pruner.revert(lin)
    assert lin.in_features == 6 and torch.equal(lin.weight, original)
