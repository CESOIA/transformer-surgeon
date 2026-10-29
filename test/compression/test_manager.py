"""compression/manager.py, scheme.py, grouping.py, smoothquant.py, cascade.py."""
import pytest
import torch
from torch.utils.data import DataLoader

from _helpers.model_factory import FAMILIES
from transformersurgeon.compression import CompressionScheme, SchemeGroup


def _model(name="qwen2"):
    torch.manual_seed(0)
    spec = FAMILIES[name]
    model = spec.build().float().eval()
    return spec, model, spec.manager(model)


def _loader(n=2):
    g = torch.Generator().manual_seed(1)
    return DataLoader([{"input_ids": torch.randint(0, 256, (12,), generator=g)} for _ in range(n)], batch_size=1)


def test_manager_builds_one_scheme_per_listed_layer_and_sentinel():
    _, model, mgr = _model()
    per_block = 4 + 3 + 2  # attention projections, MLP projections, norms
    assert len(mgr) == 2 * per_block + 3  # + embedding, head, final norm
    assert all(isinstance(s, CompressionScheme) for s in mgr)


def test_iter_filtered_criteria_grammar():
    _, _, mgr = _model()
    paths = lambda c: [s.path for s in mgr.iter_filtered(criteria=c)]
    assert paths("mlp.up_proj") == ["model.layers.0.mlp.up_proj", "model.layers.1.mlp.up_proj"]
    assert len(paths(["mlp.up_proj", "mlp.down_proj"])) == 4           # OR of items
    assert paths([["mlp.up_proj", 1]]) == ["model.layers.1.mlp.up_proj"]  # AND inside a nested list
    assert len(paths(None)) == len(mgr)


def test_set_is_declarative_and_validated():
    _, model, mgr = _model()
    mgr.set("lrd", "rank", 4, criteria="mlp.up_proj")
    assert model.model.layers[0].mlp.up_proj.rank == "full"  # nothing touched before apply
    with pytest.raises(ValueError):
        mgr.set("lrd", "rank", -1, criteria="mlp.up_proj")
    with pytest.raises(ValueError):
        mgr.set("structured_pruning", "share_mask", True, criteria="mlp.up_proj")  # group-only option


def test_groups_create_auto_and_delete():
    _, _, mgr = _model()
    groups = mgr.auto_groups()  # {group name: member paths}
    assert ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"] in groups.values()
    assert all(isinstance(g, SchemeGroup) for g in mgr.groups.values())
    name = next(iter(groups))
    mgr.delete_group(name)
    assert name not in mgr.groups
    group = SchemeGroup("g")
    group.set_property("mask", 1)
    assert group.get_property("mask") == 1 and len(group) == 0
    group.reset()
    assert group.get_property("mask") is None


def test_save_and_load_calibration_state(tmp_path):
    _, model, mgr = _model()
    mgr.set_calibration_data(_loader())
    summaries = mgr.extract_summaries(criteria="mlp.up_proj", summaries=["covariance"], show_progress=False)
    mgr.save_state(tmp_path / "state.pt")
    _, _, fresh = _model()
    fresh.load_state(tmp_path / "state.pt")
    scheme = next(iter(fresh.iter_filtered(criteria=[["mlp.up_proj", 0]])))
    torch.testing.assert_close(scheme.calibration_data["covariance"],
                               summaries["model.layers.0.mlp.up_proj"]["covariance"])


@pytest.mark.parametrize("family", ["qwen2", "llama", "mistral"])
def test_smoothquant_preserves_the_float_output(family):
    spec, model, mgr = _model(family)
    model.model.layers[0].input_layernorm.weight.data[3] *= 200.0
    ids = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        before = model(input_ids=ids).logits
    mgr.set_calibration_data(_loader())
    summary = mgr.smooth_activations(alpha=0.5)
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, before, rtol=1e-4, atol=1e-4)
    assert summary["groups"] == 3 * model.config.num_hidden_layers


def test_cascade_calibration_runs_block_by_block():
    spec, model, mgr = _model()
    mgr.set("lrd", "method", "aa-svd", criteria=spec.lrd_criteria)
    mgr.set("lrd", "rank", 8, criteria=spec.lrd_criteria)
    mgr.set_calibration_mode(mode="cascade")
    mgr.set_calibration_data(_loader())
    mgr.apply(hard=False, show_progress=False)
    assert model.model.layers[1].self_attn.q_proj.rank == 8


def test_cascade_calibration_is_refused_where_the_dictionary_forbids_it():
    spec, _, mgr = _model("bert")
    mgr.set("lrd", "method", "aa-svd", criteria=spec.lrd_criteria)
    mgr.set("lrd", "rank", 8, criteria=spec.lrd_criteria)
    mgr.set_calibration_mode(mode="cascade")
    mgr.set_calibration_data(_loader())
    with pytest.raises(ValueError, match="no_cascade_calibration"):
        mgr.apply(hard=False, show_progress=False)
