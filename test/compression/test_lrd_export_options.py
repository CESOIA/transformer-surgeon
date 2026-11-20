"""compression/: options for exporting LRD models -- runtime rank group, factor calibration of quantized
LRD layers, quantization of the inner activation, run-time SmoothQuant."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES
from transformersurgeon.blocks.layers import LinearCompressed, active_ranks
from transformersurgeon.compression.lrd import LRDer
from transformersurgeon.compression.quantization import Quantizer, export_quant_info
from transformersurgeon.compression.registry import COMPRESSION_REGISTRY


def _vit():
    torch.manual_seed(0)
    spec = FAMILIES["vit"]
    model = spec.build().eval()
    return model, spec.manager(model)


def _loader(n=8):
    torch.manual_seed(1)
    return DataLoader(TensorDataset(torch.randn(n, 3, 32, 32)), batch_size=4,
                      collate_fn=lambda b: {"pixel_values": torch.stack([t[0] for t in b])})


def _quant_config(**kw):
    cfg = {k: v["default"] for k, v in COMPRESSION_REGISTRY["quantization"].items()}
    cfg.update(kw)
    return cfg


def test_runtime_rank_group_needs_importance_ordered_factors():
    model, mgr = _vit()
    mgr.set("lrd", "rank", 16, criteria="mlp")
    mgr.set("lrd", "runtime_rank_group", "mlp", criteria="mlp")
    mgr.apply(hard=False, criteria="mlp", show_progress=False)
    fc1 = model.vit.layers[0].mlp.fc1
    assert fc1.runtime_rank_group == "mlp" and model.vit.layers[0].attention.q_proj.runtime_rank_group is None
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad(), active_ranks({"mlp": 8}):
        truncated = model(pixel_values=x).logits
    # SVD components are ordered: the first 8 of a rank-16 factorisation are the rank-8 factorisation.
    reference, ref_mgr = _vit()
    ref_mgr.set("lrd", "rank", 8, criteria="mlp")
    ref_mgr.apply(hard=False, criteria="mlp", show_progress=False)
    with torch.no_grad():
        torch.testing.assert_close(truncated, reference(pixel_values=x).logits, rtol=1e-4, atol=1e-4)
    mgr.restore(criteria="mlp")
    assert fc1.runtime_rank_group is None

    layer = LinearCompressed(8, 8)
    with pytest.raises(ValueError, match="sorted by importance"):
        LRDer({"rank": 4, "method": "aa-svd", "runtime_rank_group": "g"}).apply(layer)
    with pytest.raises(ValueError):
        mgr.set("lrd", "runtime_rank_group", "", criteria="mlp")


def _factorised(rank=16):
    torch.manual_seed(0)
    layer = LinearCompressed(64, 48, bias=False, rank=rank)
    X = torch.randn(512, 64) @ torch.randn(64, 64) * 0.3
    with torch.no_grad():
        layer.weight.normal_()
        layer.linear_V.weight.normal_()
    return layer, X, X.T @ X / X.shape[0]


@pytest.mark.parametrize("hard", [False, True])
def test_refit_calibration_reduces_the_output_error_of_quantized_factors(hard):
    errors = {}
    for mode in ("independent", "sequential", "refit"):
        layer, X, C = _factorised()
        with torch.no_grad():
            want = layer(X)
        q = Quantizer(_quant_config(method="gptq", precision=4, granularity="per_channel",
                                    lrd_factor_calibration=mode))
        q.set_calibration_store({"covariance": C})
        if mode != "independent":
            assert "covariance" in q.needs_calibration()
        q.apply(layer, hard=hard)
        with torch.no_grad():
            errors[mode] = (layer(X) - want).pow(2).mean().item()
    assert errors["refit"] < errors["independent"]
    with pytest.raises(ValueError):
        Quantizer(_quant_config(precision=4, lrd_factor_calibration="bogus")).apply(_factorised()[0])


def test_refit_works_without_gptq_and_needs_the_covariance():
    layer, X, C = _factorised()
    q = Quantizer(_quant_config(method="vanilla", precision=4, granularity="per_channel",
                                lrd_factor_calibration="refit"))
    assert q.needs_calibration() == ("covariance",)
    q.set_calibration_store({})
    with pytest.raises(RuntimeError, match="covariance"):
        q.apply(layer)


@caps.requires_cuda
def test_soft_gptq_on_lrd_takes_a_cpu_covariance():
    layer, X, C = _factorised()
    layer = layer.cuda()
    q = Quantizer(_quant_config(method="gptq", precision=8, granularity="per_channel"))
    q.set_calibration_store({"covariance": C})        # on the CPU, as with offload_to_cpu=True
    q.apply(layer, hard=False)
    assert layer.weight.device.type == "cuda"


def _w8a8(mgr, criteria, hidden=False):
    for key, value in dict(precision=8, granularity="per_channel", precision_activation=8,
                           scheme_activation="symmetric", quantize_output_activation=False).items():
        mgr.set("quantization", key, value, criteria=criteria)
    if hidden:
        mgr.set("quantization", "quantize_lrd_hidden_activation", True, criteria="mlp")


def test_inner_lrd_activation_is_calibrated_quantized_and_exported():
    model, mgr = _vit()
    mgr.set("lrd", "rank", 16, criteria="mlp")
    mgr.apply(hard=True, criteria="mlp", show_progress=False)
    mgr.set_calibration_data(_loader())
    _w8a8(mgr, ["attention", "mlp"], hidden=True)
    mgr.apply(hard=True, criteria=["attention", "mlp"], show_progress=False)
    fc2 = model.vit.layers[1].mlp.fc2
    assert hasattr(fc2.lrd_hidden, "_act_hidden_quant_scale")
    assert not hasattr(model.vit.layers[0].attention.q_proj.lrd_hidden, "_act_hidden_quant_scale")
    with torch.no_grad():
        model(pixel_values=torch.randn(1, 3, 32, 32))
    info = export_quant_info(fc2)
    assert info["act_hidden_scale"].numel() == 1 and info["act_hidden_precision"] == 8
    assert export_quant_info(model.vit.layers[0].attention.q_proj)["act_hidden_scale"] is None


def test_inner_activation_on_a_dense_layer_fails_with_a_clear_message():
    model, mgr = _vit()
    mgr.set_calibration_data(_loader())
    _w8a8(mgr, "mlp")
    mgr.set("quantization", "quantize_lrd_hidden_activation", True, criteria="mlp")
    with pytest.raises(ValueError, match="low-rank layer"):
        mgr.apply(hard=True, criteria="mlp", show_progress=False)


@pytest.mark.parametrize("lrd", [False, True])
def test_runtime_smoothing_scales_fc2_inputs_and_keeps_the_float_output(lrd):
    model, mgr = _vit()
    if lrd:
        mgr.set("lrd", "rank", 16, criteria="mlp")
        mgr.apply(hard=True, criteria="mlp", show_progress=False)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        want = model(pixel_values=x).logits
    mgr.set_calibration_data(_loader())
    summary = mgr.smooth_activations(alpha=0.5, runtime_alpha=0.8)
    assert summary["groups"] == 3 * 2 and summary["runtime_groups"] == 2 and summary["runtime_alpha"] == 0.8
    for block in model.vit.layers:
        assert block.mlp.fc2.input_scale.shape == (128,)
        assert not hasattr(block.mlp.fc1, "input_scale")
    with torch.no_grad():
        torch.testing.assert_close(model(pixel_values=x).logits, want, rtol=1e-4, atol=1e-4)

    model2, mgr2 = _vit()
    mgr2.set_calibration_data(_loader())
    assert mgr2.smooth_activations(runtime=False)["runtime_groups"] == 0
    assert not hasattr(model2.vit.layers[0].mlp.fc2, "input_scale")
