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


def test_wanda_pruning_uses_the_input_norm_summary():
    model, mgr = _model()
    mgr.set("unstructured_pruning", "method", "wanda", criteria="mlp.down_proj")
    mgr.set("unstructured_pruning", "ratio", 0.5, criteria="mlp.down_proj")
    mgr.set("unstructured_pruning", "granularity", "neuron", criteria="mlp.down_proj")
    mgr.set_calibration_data(_loader())
    mgr.apply(hard=False, show_progress=False)
    weight = model.model.layers[0].mlp.down_proj.weight
    assert ((weight == 0).sum(1) == weight.shape[1] // 2).all()  # half of every output row


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


def test_binary_qat_trains_latent_weight_and_channel_scale():
    model, mgr = _model()
    mgr.set("quantization", "precision", "binary", criteria="mlp.down_proj")
    mgr.set("quantization", "granularity", "per_channel", criteria="mlp.down_proj")
    mgr.set("quantization", "qat", True, criteria="mlp.down_proj")
    layer = model.model.layers[0].mlp.down_proj
    latent = layer.weight.detach().clone()
    mgr.apply(hard=False)
    assert layer.bin_scale.shape == (layer.out_features, 1) and layer.bin_scale.requires_grad
    torch.testing.assert_close(layer.weight, latent)  # latent weight untouched
    torch.testing.assert_close(layer.bin_scale.detach(), latent.abs().mean(dim=1, keepdim=True))
    model.train()
    model(input_ids=IDS).logits.sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0  # STE passes the gradient
    assert layer.bin_scale.grad is not None and layer.bin_scale.grad.abs().sum() > 0
    x = torch.randn(3, layer.in_features)
    expected = x @ (layer.bin_scale.detach() * torch.where(latent >= 0, 1.0, -1.0)).T + layer.bias.detach() if layer.bias is not None else x @ (layer.bin_scale.detach() * torch.where(latent >= 0, 1.0, -1.0)).T
    torch.testing.assert_close(layer(x), expected)
    mgr.restore()
    assert layer.bin_scale is None


def test_binary_qat_requires_binary_precision():
    model, mgr = _model()
    mgr.set("quantization", "precision", 8, criteria="mlp.down_proj")
    mgr.set("quantization", "qat", True, criteria="mlp.down_proj")
    with pytest.raises(NotImplementedError):
        mgr.apply(hard=False)


def test_gptq_binary_qat_uses_the_covariance_summary():
    model, mgr = _model()
    crit = "mlp.down_proj"
    mgr.set("quantization", "method", "gptq", criteria=crit)
    mgr.set("quantization", "precision", "binary", criteria=crit)
    mgr.set("quantization", "granularity", "per_channel", criteria=crit)
    mgr.set("quantization", "qat", True, criteria=crit)
    mgr.set_calibration_data(_loader())
    layer = model.model.layers[0].mlp.down_proj
    latent = layer.weight.detach().clone()
    mgr.apply(hard=False, show_progress=False)
    assert layer.bin_scale is not None
    assert not torch.allclose(layer.weight, latent)  # Hessian compensation moved the latent weight
    torch.testing.assert_close(layer.bin_scale.detach(), latent.abs().mean(dim=1, keepdim=True))


def _flap_setup(ratio=0.5, bias_compensation=True):
    model, mgr = _model()
    mgr.set("channel_pruning", "ratio", ratio, criteria="mlp.down_proj")
    mgr.set("channel_pruning", "bias_compensation", bias_compensation, criteria="mlp.down_proj")
    mgr.set_calibration_data(_loader(4))
    return model, mgr


def test_flap_prunes_the_requested_fraction_globally_and_adds_the_baseline_bias():
    model, mgr = _flap_setup(0.5)
    dense = _logits(model)
    mgr.apply(hard=False, show_progress=False)
    layers = [blk.mlp.down_proj for blk in model.model.layers]
    kept = sum(int(l.input_mask.sum()) for l in layers)
    total = sum(l.in_features for l in layers)
    assert abs(kept / total - 0.5) < 0.02
    for l in layers:
        assert (l.weight[:, ~l.input_mask] == 0).all()
        assert l.bias is not None
    assert not torch.allclose(_logits(model), dense)
    mgr.restore()
    assert all(blk.mlp.down_proj.bias is None for blk in model.model.layers)
    torch.testing.assert_close(_logits(model), dense)


def test_flap_bias_compensation_improves_over_plain_zeroing():
    errs = {}
    for comp in (True, False):
        model, mgr = _flap_setup(0.5, comp)
        dense = _logits(model)
        mgr.apply(hard=False, show_progress=False)
        errs[comp] = (_logits(model) - dense).pow(2).mean().item()
    assert errs[True] <= errs[False]


def test_flap_hard_matches_soft_and_shrinks_the_mlp():
    model, mgr = _flap_setup(0.5)
    mgr.apply(hard=False, show_progress=False)
    soft = _logits(model)
    mgr.apply(hard=True, show_progress=False)
    layer = model.model.layers[0].mlp
    assert layer.down_proj.in_features == layer.gate_proj.out_features == layer.up_proj.out_features
    assert layer.down_proj.in_features < 128
    torch.testing.assert_close(_logits(model), soft, rtol=1e-4, atol=1e-4)


def test_flap_groups_prune_whole_heads_across_attention_and_mlp():
    model, mgr = _model()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    mgr.set("channel_pruning", "ratio", 0.4, criteria=["mlp.down_proj", "self_attn.o_proj"])
    mgr.set("channel_pruning", "group_size", head_dim, criteria="self_attn.o_proj")
    mgr.set("channel_pruning", "rank_group", "attn", criteria="self_attn.o_proj")
    mgr.set("channel_pruning", "rank_group", "mlp", criteria="mlp.down_proj")
    mgr.set_calibration_data(_loader(4))
    dense = _logits(model)
    mgr.apply(hard=False, show_progress=False)
    for blk in model.model.layers:
        m = blk.self_attn.o_proj.input_mask.view(-1, head_dim)
        assert (m.all(1) | (~m).all(1)).all()  # a head is kept or dropped as a whole
    cost = lambda l: l.in_features * l.out_features
    pruned = sum(int((~b.mlp.down_proj.input_mask).sum()) * b.mlp.down_proj.out_features + int((~b.self_attn.o_proj.input_mask).sum()) * b.self_attn.o_proj.out_features for b in model.model.layers)
    total = sum(cost(b.mlp.down_proj) + cost(b.self_attn.o_proj) for b in model.model.layers)
    assert abs(pruned / total - 0.4) < 0.05
    attn = [b.self_attn.o_proj for b in model.model.layers]
    frac = sum(int((~l.input_mask).sum()) for l in attn) / sum(l.in_features for l in attn)
    assert abs(frac - 0.4) < 0.15  # each pool has its own threshold for the same ratio
    assert not torch.allclose(_logits(model), dense)
