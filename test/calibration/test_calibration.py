"""calibration/: raw-data collectors, summaries, the generic runner and the standalone absmax pass."""
import pytest
import torch
from torch.utils.data import DataLoader

from _helpers.model_factory import FAMILIES
from transformersurgeon.calibration import SUPPORTED_SUMMARIES, collect_input_absmax
from transformersurgeon.calibration.raw_data.registry import RAW_DATA_REGISTRY
from transformersurgeon.calibration.summaries.registry import SUMMARY_REGISTRY, get_summary

LAYER = "model.layers.0.mlp.down_proj"


def _setup():
    torch.manual_seed(0)
    spec = FAMILIES["qwen2"]
    model = spec.build().float().eval()
    g = torch.Generator().manual_seed(1)
    data = [{"input_ids": torch.randint(0, 256, (6,), generator=g)} for _ in range(3)]
    return model, spec.manager(model), DataLoader(data, batch_size=1)


def _captured_io(model, loader):
    """Inputs and outputs of LAYER over the loader, captured with a plain hook."""
    xs, ys = [], []
    layer = model.get_submodule(LAYER)

    def hook(module, inputs, output):
        xs.append(inputs[0].reshape(-1, inputs[0].shape[-1]))
        ys.append(output.reshape(-1, output.shape[-1]))

    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        for batch in loader:
            model(**batch)
    handle.remove()
    return torch.cat(xs), torch.cat(ys)


def test_registries_are_consistent():
    assert set(SUPPORTED_SUMMARIES) == set(SUMMARY_REGISTRY)
    for name in SUPPORTED_SUMMARIES:
        assert set(get_summary(name).required_raw_data) <= set(RAW_DATA_REGISTRY)
    with pytest.raises(ValueError):
        get_summary("median")


def test_summaries_match_direct_computation():
    model, mgr, loader = _setup()
    x, y = _captured_io(model, loader)
    mgr.set_calibration_data(loader)
    out = mgr.extract_summaries(criteria=[["mlp.down_proj", 0]], show_progress=False, summaries=[
        "covariance", "activation_range", "output_activation_range", "weight",
        "input_activation", "output_activation"])[LAYER]
    torch.testing.assert_close(out["covariance"], x.T @ x / len(x), rtol=1e-4, atol=1e-5)
    assert torch.isclose(out["activation_range"]["min"], x.min()) and torch.isclose(out["activation_range"]["max"], x.max())
    assert torch.isclose(out["output_activation_range"]["max"], y.max())
    torch.testing.assert_close(out["weight"], model.get_submodule(LAYER).weight)
    torch.testing.assert_close(out["input_activation"], x)
    torch.testing.assert_close(out["output_activation"], y)


def test_bias_summary_requires_a_bias():
    model, mgr, loader = _setup()
    mgr.set_calibration_data(loader)
    with pytest.raises(ValueError, match="has no bias"):
        mgr.extract_summaries(criteria=[["mlp.down_proj", 0]], summaries=["bias"], show_progress=False)
    q = mgr.extract_summaries(criteria=[["self_attn.q_proj", 0]], summaries=["bias"], show_progress=False)
    torch.testing.assert_close(q["model.layers.0.self_attn.q_proj"]["bias"], model.model.layers[0].self_attn.q_proj.bias)


def test_weight_gradient_summary_runs_backward():
    model, mgr, loader = _setup()
    mgr.set_calibration_data(loader)
    mgr.set_calibration_loss(lambda output, label: output.logits.float().pow(2).mean())
    out = mgr.extract_summaries(criteria=[["mlp.down_proj", 0]], summaries=["weight_grad"], show_progress=False)
    grad = out[LAYER]["weight_grad"]
    assert grad.shape == model.get_submodule(LAYER).weight.shape and grad.abs().sum() > 0


def test_collect_input_absmax_is_per_channel_max_abs():
    model, _, loader = _setup()
    x, _ = _captured_io(model, loader)
    layer = model.get_submodule(LAYER)
    stats = collect_input_absmax(model, [layer], loader)
    torch.testing.assert_close(stats[layer], x.abs().amax(0))


def test_token_windows_loader_random_and_sequential():
    from transformersurgeon.calibration import token_windows_loader

    ids = list(range(100))
    random_windows = [b["input_ids"][0] for b in token_windows_loader(ids, 5, 10, seed=3)]
    assert len(random_windows) == 5 and all(w.tolist() == list(range(w[0], w[0] + 10)) for w in random_windows)
    assert [b["input_ids"][0].tolist() for b in token_windows_loader(ids, 5, 10, seed=3)] == \
           [w.tolist() for w in random_windows]  # seeded
    sequential = list(token_windows_loader(ids, 50, 30, sequential=True, labels=True))
    assert [b["input_ids"][0, 0].item() for b in sequential] == [0, 30, 60]  # at most len // seq_len
    assert all(torch.equal(b["labels"], b["input_ids"]) and b["attention_mask"].sum() == 30 for b in sequential)
