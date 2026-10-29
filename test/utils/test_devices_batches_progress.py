"""utils/devices.py, utils/batches.py, utils/progress.py."""
import pytest
import torch

from transformersurgeon.utils import (
    FallbackProgressBar,
    build_progress_bar,
    infer_model_device,
    move_to_device,
    normalize_batch,
    pack_batch,
)


def test_infer_model_device_from_parameters_buffers_and_fallback():
    assert infer_model_device(torch.nn.Linear(2, 2)) == torch.device("cpu")
    buffered = torch.nn.Module()
    buffered.register_buffer("b", torch.zeros(1))
    assert infer_model_device(buffered) == torch.device("cpu")
    assert infer_model_device(torch.nn.Module()) == torch.device("cpu")


def test_move_to_device_recurses_into_containers():
    value = {"a": torch.zeros(1), "b": [torch.ones(1), (torch.ones(2), 3)], "c": "text"}
    moved = move_to_device(value, torch.device("cpu"))
    assert isinstance(moved["b"][1], tuple) and moved["b"][1][1] == 3
    assert moved["c"] == "text"
    assert torch.equal(moved["b"][0], torch.ones(1))


@pytest.mark.parametrize("batch, expected", [
    ({"input_ids": 1}, ((), {"input_ids": 1}, None)),
    (({"input_ids": 1}, "label"), ((), {"input_ids": 1}, "label")),
    (("label", {"input_ids": 1}), ((), {"input_ids": 1}, "label")),
    ((1, 2, 3), ((1, 2, 3), {}, None)),
    ([{"x": 1}], ((), {"x": 1}, None)),
    ("single", (("single",), {}, None)),
])
def test_normalize_batch_forms(batch, expected):
    assert normalize_batch(batch, batch_id=0) == expected


def test_pack_batch_round_trips_through_normalize():
    assert normalize_batch(pack_batch((1,), {"k": 2}, "y"), batch_id=0) == ((1,), {"k": 2}, "y")
    assert normalize_batch(pack_batch([1, 2]), batch_id=0) == ((1, 2), {}, None)


def test_normalize_batch_rejects_empty_and_bad_kwargs():
    with pytest.raises(ValueError):
        normalize_batch((), batch_id=3)
    with pytest.raises(TypeError):
        normalize_batch({"__ts_args__": (), "__ts_kwargs__": [1]}, batch_id=0)


def test_progress_bars(capsys):
    bar = FallbackProgressBar(total=2, enabled=True, desc="Work")
    bar.update()
    bar.close()
    assert "Work: 1/2" in capsys.readouterr().out
    disabled = build_progress_bar(enabled=False, total=3, verbose=True)
    assert isinstance(disabled, FallbackProgressBar) and not disabled.enabled
