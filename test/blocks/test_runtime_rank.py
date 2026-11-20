"""blocks/layers: runtime-selectable rank, the inner-activation hook point and the run-time input scale."""
import pytest
import torch

from transformersurgeon.blocks.layers import LinearCompressed, active_rank, active_ranks, runtime_rank_groups


def _lrd_layer(rank=6, group="g", bias=True):
    torch.manual_seed(0)
    layer = LinearCompressed(12, 10, bias=bias, rank=rank, runtime_rank_group=group)
    with torch.no_grad():
        layer.weight.normal_()
        layer.linear_V.weight.normal_()
    return layer


def _truncated(layer, x, r):
    return torch.nn.functional.linear(x @ layer.linear_V.weight[:r].T, layer.weight[:, :r], layer.bias)


def test_active_rank_runs_the_first_components():
    layer = _lrd_layer()
    x = torch.randn(3, 5, 12)
    with torch.no_grad():
        full = layer(x)
        with active_ranks({"g": 4}):
            torch.testing.assert_close(layer(x), _truncated(layer, x, 4))
        with active_ranks({"g": 99}):            # clamped to the stored rank
            torch.testing.assert_close(layer(x), full)
        with active_ranks({"other": 2}):         # another group: stored rank
            torch.testing.assert_close(layer(x), full)
        with active_ranks({"g": torch.zeros(3)}):   # a tensor rank is its length
            torch.testing.assert_close(layer(x), _truncated(layer, x, 3))
    assert active_rank("g") is None


def test_active_ranks_nest_and_validate():
    with active_ranks({"a": 4, "b": 2}):
        with active_ranks({"a": 3}):
            assert (active_rank("a"), active_rank("b")) == (3, 2)
        assert active_rank("a") == 4
    for bad in (0, -1, 2.5, True, torch.zeros(2, 2)):
        with pytest.raises(ValueError):
            with active_ranks({"a": bad}):
                pass


def test_layers_without_a_group_ignore_the_context():
    layer = _lrd_layer(group=None)
    x = torch.randn(4, 12)
    with torch.no_grad():
        want = layer(x)
        with active_ranks({"g": 2}):
            torch.testing.assert_close(layer(x), want)


def test_runtime_rank_groups_lists_lrd_layers_by_group():
    net = torch.nn.ModuleDict({"a": _lrd_layer(group="mlp"), "b": _lrd_layer(group="mlp"),
                               "c": _lrd_layer(group="attn"), "d": LinearCompressed(4, 4)})
    groups = runtime_rank_groups(net)
    assert list(groups) == ["attn", "mlp"]
    assert [name for name, _ in groups["mlp"]] == ["a", "b"]


def test_lrd_hidden_is_the_hook_point_between_the_factors():
    layer = _lrd_layer()
    x = torch.randn(2, 12)
    seen = {}
    layer.lrd_hidden.register_forward_hook(lambda m, i, o: seen.setdefault("h", o))
    with torch.no_grad():
        layer(x)
        torch.testing.assert_close(seen["h"], x @ layer.linear_V.weight.T)
    assert isinstance(LinearCompressed(4, 4).lrd_hidden, torch.nn.Identity)


def test_input_scale_runs_first_and_survives_a_state_dict_load():
    layer = _lrd_layer(group=None)
    s = torch.rand(12) + 0.5
    x = torch.randn(3, 12)
    with torch.no_grad():
        want = layer(x * s)
    layer.register_forward_pre_hook(lambda m, i: seen.__setitem__("x", i[0]))   # registered before the scale
    seen = {}
    layer.set_input_scale(s)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), want)
    torch.testing.assert_close(seen["x"], x * s)   # other pre-hooks see the scaled input

    fresh = LinearCompressed(12, 10, bias=True, rank=6)
    fresh.load_state_dict(layer.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(fresh(x), want)

    layer.set_input_scale(None)
    assert "input_scale" not in dict(layer.named_buffers())
    with pytest.raises(ValueError, match="channels"):
        layer.set_input_scale(torch.ones(5))
