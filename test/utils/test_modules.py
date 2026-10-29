"""utils/modules.py: dotted-path access and replacement of submodules and parameters."""
import pytest
import torch

from transformersurgeon.utils import get_parent_module, get_submodule, replace_parameter, set_submodule


def _net():
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(3, 2)))


def test_get_submodule_resolves_nested_paths_and_list_indices():
    net = _net()
    assert get_submodule(net, "1.1") is net[1][1]
    assert get_submodule(net, "0") is net[0]


def test_get_submodule_raises_on_missing_path():
    with pytest.raises(ValueError, match="not found"):
        get_submodule(_net(), "1.7")


def test_get_parent_module_splits_last_component():
    net = _net()
    assert get_parent_module(net, "1.1") == (net[1], "1")
    assert get_parent_module(net, "0") == (net, "0")


def test_set_submodule_replaces_in_place():
    net = _net()
    new = torch.nn.Identity()
    set_submodule(net, "1.0", new)
    assert net[1][0] is new


def test_replace_parameter_changes_shape_and_keeps_requires_grad():
    lin = torch.nn.Linear(4, 3)
    lin.weight.requires_grad_(False)
    replace_parameter(lin, "weight", torch.zeros(2, 4))
    assert isinstance(lin.weight, torch.nn.Parameter)
    assert lin.weight.shape == (2, 4)
    assert lin.weight.requires_grad is False
