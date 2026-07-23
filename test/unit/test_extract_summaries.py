"""manager.extract_summaries(): summary-agnostic calibration extraction.

Covers the new 'weight'/'bias'/'input_activation'/'output_activation' summaries
added alongside the existing calibration summaries, and the generic
extract_summaries() entrypoint that runs calibration and returns
{layer_path: {summary_name: value}} without special-casing any summary name.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from _helpers.model_factory import FAMILIES

pytestmark = pytest.mark.unit


def _tiny_calibration_loader(seq_len=8, num_batches=3):
    examples = [{"input_ids": torch.randint(0, 256, (seq_len,))} for _ in range(num_batches)]
    return DataLoader(examples, batch_size=1, shuffle=False)


def test_extract_summaries_returns_nested_dict_for_all_kinds():
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    mgr.set_calibration_data(_tiny_calibration_loader(num_batches=3))

    data = mgr.extract_summaries(
        criteria=spec.lrd_criteria,
        summaries=["weight", "bias", "input_activation", "output_activation", "covariance"],
        show_progress=False,
    )

    assert len(data) > 0
    for layer_path, layer_summaries in data.items():
        module = mgr._find_scheme(layer_path).get_compression_module()

        assert torch.equal(layer_summaries["weight"], module.weight.detach().float())

        if module.bias is not None:
            assert torch.equal(layer_summaries["bias"], module.bias.detach().float())
        else:
            assert layer_summaries["bias"] is None

        # 3 batches of seq_len=8 tokens each -> 24 tokens concatenated.
        assert layer_summaries["input_activation"].shape[0] == 24
        assert layer_summaries["output_activation"].shape[0] == 24
        assert layer_summaries["input_activation"].shape[-1] == module.in_features
        assert layer_summaries["output_activation"].shape[-1] == module.out_features

        assert layer_summaries["covariance"].shape == (module.in_features, module.in_features)


def test_extract_summaries_bias_raises_clear_error_when_module_has_no_bias():
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    mgr.set_calibration_data(_tiny_calibration_loader(num_batches=1))

    no_bias_layers = [
        scheme.path for scheme in mgr.iter_filtered(criteria=spec.mlp_prune_criteria)
        if scheme.get_compression_module().bias is None
    ]
    if not no_bias_layers:
        pytest.skip("No bias-less layer available for this family/criteria.")

    with pytest.raises(ValueError, match="bias"):
        mgr.extract_summaries(criteria=no_bias_layers[0], summaries=["bias"], show_progress=False)


def test_extract_summaries_requires_nonempty_summaries():
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)

    with pytest.raises(ValueError, match="non-empty"):
        mgr.extract_summaries(criteria=spec.lrd_criteria, summaries=[])


def test_extract_summaries_rejects_unknown_summary_name():
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    mgr.set_calibration_data(_tiny_calibration_loader(num_batches=1))

    with pytest.raises(ValueError, match="Unsupported summaries"):
        mgr.extract_summaries(criteria=spec.lrd_criteria, summaries=["not_a_real_summary"], show_progress=False)
