"""SmoothQuant (compression/smoothquant.py, manager.smooth_activations) on tiny random models."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from _helpers.model_factory import FAMILIES


def _loader(n=4, seq=12):
    g = torch.Generator().manual_seed(1)
    return DataLoader([{"input_ids": torch.randint(0, 256, (seq,), generator=g)} for _ in range(n)], batch_size=1)


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids[None]).logits[0].float()


@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_smoothing_preserves_float_output_and_tames_outliers(family):
    spec = FAMILIES[family]
    torch.manual_seed(0)
    model = spec.build().float().eval()
    blk = model.model.layers[0]
    blk.input_layernorm.weight.data[3] *= 200.0    # outlier channel into q/k/v
    blk.mlp.up_proj.weight.data[5] *= 200.0         # outlier channel into down_proj
    ids = torch.randint(0, 256, (10,))
    before = _logits(model, ids)

    manager = spec.manager(model)
    manager.set_calibration_data(_loader())
    summary = manager.smooth_activations(alpha=0.5)

    torch.testing.assert_close(_logits(model, ids), before, rtol=1e-4, atol=1e-4)
    n_layers = model.config.num_hidden_layers
    assert summary["groups"] == 3 * n_layers
    for key in ("self_attn.q_proj", "mlp.down_proj"):
        assert summary["max_input_absmax_after"][key] < 0.2 * summary["max_input_absmax_before"][key]


def test_smoothing_criteria_selects_whole_groups_only():
    spec = FAMILIES["qwen2"]
    torch.manual_seed(0)
    model = spec.build().float().eval()
    q = model.model.layers[0].self_attn.q_proj.weight.clone()
    norm = model.model.layers[0].input_layernorm.weight.clone()
    manager = spec.manager(model)
    manager.set_calibration_data(_loader())
    summary = manager.smooth_activations(alpha=0.5, criteria=["mlp"])
    assert summary["groups"] == 2 * model.config.num_hidden_layers  # gate/up and down, not q/k/v
    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight, q)
    assert torch.equal(model.model.layers[0].input_layernorm.weight, norm)
    # q_proj alone selects nothing: its producer also feeds k/v
    m2 = spec.manager(model)
    m2.set_calibration_data(_loader())
    assert m2.smooth_activations(criteria=["q_proj"])["groups"] == 0


def test_smoothing_refuses_quantized_weights():
    spec = FAMILIES["qwen2"]
    torch.manual_seed(0)
    model = spec.build().float().eval()
    manager = spec.manager(model)
    manager.set("quantization", "precision", 8, criteria=["mlp"])
    manager.set("quantization", "granularity", "per_channel", criteria=["mlp"])
    manager.apply(hard=True, criteria=["mlp"])
    manager.set_calibration_data(_loader())
    with pytest.raises(RuntimeError, match="already quantized"):
        manager.smooth_activations(criteria=["mlp"])


def test_smoothing_requires_calibration_data():
    spec = FAMILIES["qwen2"]
    model = spec.build().float().eval()
    with pytest.raises(ValueError, match="calibration data"):
        spec.manager(model).smooth_activations()
