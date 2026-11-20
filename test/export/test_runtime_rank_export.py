"""Encoder export of LRD models: runtime rank inputs, slice-aware Q/DQ, inner-activation Q/DQ, TensorRT
rank profiles and TensorRTEncoderSession."""
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES
from transformersurgeon.blocks.layers import active_ranks
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.runtime_ranks import rank_input_name, resolve_runtime_ranks

RANK = 16


def _loader():
    torch.manual_seed(1)
    return DataLoader(TensorDataset(torch.randn(8, 3, 32, 32)), batch_size=4,
                      collate_fn=lambda b: {"pixel_values": torch.stack([t[0] for t in b])})


def _model(runtime=True, w8a8=False):
    torch.manual_seed(0)
    spec = FAMILIES["vit"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    mgr.set("lrd", "rank", RANK, criteria="mlp")
    if runtime:
        mgr.set("lrd", "runtime_rank_group", "mlp", criteria="mlp")
    mgr.apply(hard=True, criteria="mlp", show_progress=False)
    if w8a8:
        mgr.set_calibration_data(_loader())
        mgr.smooth_activations(alpha=0.5)
        for key, value in dict(precision=8, granularity="per_channel", precision_activation=8,
                               scheme_activation="symmetric", quantize_output_activation=False).items():
            mgr.set("quantization", key, value, criteria=["attention", "mlp"])
        mgr.set("quantization", "quantize_lrd_hidden_activation", True, criteria="mlp")
        mgr.apply(hard=True, criteria=["attention", "mlp"], show_progress=False)
    return model


def test_resolve_runtime_ranks_validates_groups_and_ranges():
    model = _model()
    ranks = resolve_runtime_ranks(model)
    assert list(ranks) == ["mlp"] and ranks["mlp"]["input"] == rank_input_name("mlp") == "rank_mlp"
    assert (ranks["mlp"]["min"], ranks["mlp"]["max"], len(ranks["mlp"]["layers"])) == (1, RANK, 4)
    assert resolve_runtime_ranks(model, {}) == {}
    assert resolve_runtime_ranks(model, {"mlp": (4, 8, 12)})["mlp"]["opt"] == 8
    with pytest.raises(ValueError, match="not found"):
        resolve_runtime_ranks(model, {"attn": None})
    with pytest.raises(ValueError, match="stored rank"):
        resolve_runtime_ranks(model, {"mlp": (4, 8, RANK + 1)})
    model.vit.layers[0].mlp.fc1.init_lrd(8)
    with pytest.raises(ValueError, match="mixes stored ranks"):
        resolve_runtime_ranks(model)


@caps.requires_onnx
def test_onnx_encoder_export_takes_a_rank_input(tmp_path):
    import onnx
    from transformersurgeon.export.backends import ONNXExportConfig

    result = export_to_backend(_model(), ONNXExportConfig(output_path=str(tmp_path / "model.onnx"),
                                                          runtime_ranks={"mlp": (2, 8, RANK)}))
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    assert [i.name for i in graph.input] == ["pixel_values", "rank_mlp"]
    assert graph.input[1].type.tensor_type.shape.dim[0].dim_param      # dynamic length
    manifest = json.load(open(result.manifest_path))
    assert manifest["runtime_ranks"]["mlp"]["max"] == RANK
    assert manifest["inputs"]["rank_mlp"] == {"shape": ["rank_mlp"], "dtype": "int32", "runtime_rank_group": "mlp"}
    assert sum(n.op_type == "Slice" for n in graph.node) >= 8           # both factors of 4 layers

    static = export_to_backend(_model(), ONNXExportConfig(output_path=str(tmp_path / "s.onnx"), runtime_ranks={}))
    assert [i.name for i in onnx.load(static.onnx_path, load_external_data=False).graph.input] == ["pixel_values"]


@caps.requires_onnx
def test_w8a8_lrd_export_quantizes_both_factor_inputs_and_rescales_sliced_rows(tmp_path):
    import onnx
    from transformersurgeon.export.backends import ONNXExportConfig

    result = export_to_backend(_model(w8a8=True), ONNXExportConfig(output_path=str(tmp_path / "model.onnx")))
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    manifest = json.load(open(result.manifest_path))
    fc2 = manifest["quantized_layers"]["encoder.blocks.0.mlp.down_proj"]
    assert fc2["act_hidden_precision"] == 8
    assert manifest["quantized_layers"]["encoder.blocks.0.mlp.down_proj.linear_V"]["granularity"] == \
        "per_channel_row_sliced"
    names = {n.name for n in graph.node}
    assert "encoder.blocks.0.mlp.down_proj.act_hidden0_QuantizeLinear" in names
    assert "encoder.blocks.0.mlp.down_proj.linear_V.weight_row_scale_0" in names
    # The run-time SmoothQuant scale of fc2 (down_proj) is a Mul in the graph.
    assert any(n.op_type == "Mul" and any("input_scale" in i for i in n.input) for n in graph.node)


@caps.requires_tensorrt
@pytest.mark.parametrize("w8a8", [False, True])
def test_tensorrt_rank_profiles_and_session_match_the_eager_model(tmp_path, w8a8):
    from transformersurgeon.export.backends import TensorRTExportConfig
    from transformersurgeon.export.backends.tensorrt import TensorRTEncoderSession, engine_profiles

    model = _model(w8a8=w8a8)
    x = torch.randn(1, 3, 32, 32)
    want = {}
    with torch.no_grad():
        for r in (4, 8, 12, RANK):
            with active_ranks({"mlp": r}):
                want[r] = model(pixel_values=x).logits
    result = export_to_backend(model, TensorRTExportConfig(
        output_path=str(tmp_path / "model.onnx"), float_type=torch.float16, fp32_accumulate_lrd=False,
        runtime_ranks={"mlp": (2, RANK, RANK)}, rank_profiles=[{"mlp": 8}, {"mlp": RANK}, {"mlp": (2, 8, RANK)}]))
    manifest = json.load(open(result.manifest_path))
    assert manifest["rank_profiles"] == [{"mlp": [8, 8, 8]}, {"mlp": [RANK] * 3}, {"mlp": [2, 8, RANK]}]
    assert len(engine_profiles(manifest)) == 3

    session = TensorRTEncoderSession(result.manifest_path)
    assert session.ranks == {"mlp": RANK} and session.rank_range == {"mlp": (2, RANK)}
    tol = 0.25 if w8a8 else 2e-2   # the engine quantizes in fp16 and rounds differently at boundaries
    for r, profile in ((8, 0), (12, 2), (RANK, 1), (4, 2), (8, 0)):   # back and forth: graphs stay valid
        out = session(x.half().cuda(), {"mlp": r}).float().cpu()
        assert session.profile == profile
        torch.testing.assert_close(out, want[r], rtol=tol, atol=tol)
    with pytest.raises(ValueError, match="profile"):
        session.set_ranks({"mlp": 1})
    with pytest.raises(KeyError):
        session.set_ranks({"attn": 4})
