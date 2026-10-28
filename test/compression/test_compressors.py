"""Compressors (LRDer, StructuredPruner, UnstructuredPruner, BlockPruner, Quantizer) applied through the manager."""
import pytest
import torch
from torch.utils.data import DataLoader

from _helpers.model_factory import FAMILIES

IDS = (torch.arange(1, 13) * 7 % 256)[None]


def _model(name="qwen2"):
    torch.manual_seed(0)
    spec = FAMILIES[name]
    model = spec.build().eval()
    return model, spec.manager(model)


def _logits(model):
    with torch.no_grad():
        return model(input_ids=IDS).logits


def _loader(n=2):
    g = torch.Generator().manual_seed(1)
    return DataLoader([{"input_ids": torch.randint(0, 256, (12,), generator=g)} for _ in range(n)], batch_size=1)


def test_lrd_soft_apply_and_restore():
    model, mgr = _model()
    before = _logits(model)
    mgr.set("lrd", "rank", 4, criteria="mlp.down_proj")
    mgr.apply(hard=False)
    layer = model.model.layers[0].mlp.down_proj
    assert layer.rank == 4 and layer.weight.shape == (64, 4)
    assert not torch.allclose(_logits(model), before)
    mgr.restore()
    assert layer.rank == "full"
    torch.testing.assert_close(_logits(model), before)


def test_calibrated_lrd_uses_the_covariance_summary():
    model, mgr = _model()
    mgr.set("lrd", "method", "svd-llm-v2", criteria="mlp.down_proj")
    mgr.set("lrd", "rank", 8, criteria="mlp.down_proj")
    mgr.set_calibration_data(_loader())
    mgr.apply(hard=False, show_progress=False)
    assert model.model.layers[1].mlp.down_proj.rank == 8


def test_structured_pruning_soft_zeroes_and_hard_removes_rows():
    model, mgr = _model()
    for name, paths in mgr.auto_groups().items():
        if any("gate_proj" in path for path in paths):
            mgr.set("structured_pruning", "share_mask", True, group=name)
            mgr.set("structured_pruning", "reduce_op", "add", group=name)
    mgr.set("structured_pruning", "ratio", 0.25, criteria=["mlp.gate_proj", "mlp.up_proj"])
    mgr.apply(hard=False)
    gate = model.model.layers[0].mlp.gate_proj
    assert (gate.weight.abs().sum(1) == 0).sum() == 32
    soft = _logits(model)
    mgr.apply(hard=True)
    assert gate.out_features == 96 and model.model.layers[0].mlp.down_proj.in_features == 96
    torch.testing.assert_close(_logits(model), soft, rtol=1e-4, atol=1e-4)


def test_unstructured_pruning_masks_weights():
    model, mgr = _model()
    mgr.set("unstructured_pruning", "ratio", 0.5, criteria="mlp.up_proj")
    mgr.apply(hard=False)
    w = model.model.layers[0].mlp.up_proj.weight
    assert (w == 0).float().mean() >= 0.5


def test_block_pruning_removes_whole_heads():
    model, mgr = _model()
    soft_before = _logits(model)
    # 4 query heads over 2 KV heads: remove one head from each KV group.
    mgr.set("block_pruning", "blocks", [1, 3], criteria=[["self_attn.q_proj", 0]])
    mgr.apply(hard=False)
    soft = _logits(model)
    assert not torch.allclose(soft, soft_before)
    mgr.apply(hard=True)
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.out_features == 2 * 16 and attn.o_proj.in_features == 2 * 16
    torch.testing.assert_close(_logits(model), soft, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["vanilla", "hqq"])
def test_quantizer_soft_and_hard_tags(method):
    model, mgr = _model()
    before = _logits(model)
    mgr.set("quantization", "method", method, criteria="mlp")
    mgr.set("quantization", "precision", 8, criteria="mlp")
    mgr.set("quantization", "granularity", "per_channel", criteria="mlp")
    mgr.apply(hard=False)
    layer = model.model.layers[0].mlp.up_proj
    assert layer._soft_quant_precision == 8
    torch.testing.assert_close(_logits(model), before, rtol=0.05, atol=0.05)
    mgr.apply(hard=True)
    assert layer._torchao_precision == 8 and layer._torchao_scale.numel() == layer.out_features
    mgr.prepare_for_save()
    assert not hasattr(layer, "_torchao_scale") and type(layer.weight) is torch.nn.Parameter


def test_static_activation_quantisation_installs_and_strips_hooks():
    model, mgr = _model()
    mgr.set("quantization", "precision", 8, criteria="mlp.down_proj")
    mgr.set("quantization", "precision_activation", 8, criteria="mlp.down_proj")
    mgr.set_calibration_data(_loader())
    mgr.apply(hard=False, show_progress=False)
    layer = model.model.layers[0].mlp.down_proj
    assert layer._act_quant_precision == 8 and hasattr(layer, "_act_quant_hook_handle")
    _logits(model)
    mgr.prepare_for_save()
    assert not hasattr(layer, "_act_quant_hook_handle") and not hasattr(layer, "_act_quant_scale")
