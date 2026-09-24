"""ONNX export ("onnx") and plain-TensorRT ("tensorrt_onnx") backends on a tiny
random-weight Qwen2 -- no downloads. TensorRT tests are capability-gated.

See docs/investigations/TENSORRT_ONNX_EXPORT.md for why this path exists.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from _helpers import capabilities as caps
from _helpers.model_factory import _qwen2

MAX_CACHE = 16
CONVERT = {"use_sdpa": False, "cache_impl": "io_inplace", "max_cache_len": MAX_CACHE,
           "rmsnorm_prescale": False, "rmsnorm_upcast": True}


def _tiny(dtype=torch.float32):
    torch.manual_seed(0)
    model = _qwen2().to(dtype).eval()
    model.config.dtype = dtype  # convert_for_export sizes the decoder from the config dtype
    return model


def _eager_logits(model, token_ids, device):
    """Reference: the converted io_inplace graph in eager PyTorch, prompt in one call."""
    from transformersurgeon.export.common import build_wrapper, build_zero_caches
    from transformersurgeon.utils import convert_for_export

    decoder = convert_for_export(model, options=CONVERT)["text"]
    wrapper = build_wrapper({"embedding": model.get_input_embeddings(), "decoder": decoder,
                             "final_layer": model.lm_head}, model_config=model.config).eval().to(device)
    keys, values = build_zero_caches(wrapper.decoder)
    with torch.no_grad():
        return wrapper(token_ids.to(device), torch.tensor([0], device=device),
                       [k.to(device) for k in keys], [v.to(device) for v in values])[0].float()


@caps.requires_onnx
def test_onnx_export_io_contract_and_manifest(tmp_path):
    import onnx

    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.onnx import ONNXExportConfig

    path = str(tmp_path / "model.onnx")
    cfg = ONNXExportConfig(output_path=path, backend="onnx", max_input_len=8, convert_options=CONVERT)
    result = export_to_backend(_tiny(), config=cfg)

    model = onnx.load(path, load_external_data=False)
    ops = [n.op_type for n in model.graph.node]
    assert ops.count("TensorScatter") == 4  # 2 layers x (key, value), in-place KV writes
    assert [i.name for i in model.graph.input][:4] == ["input_ids", "pos_id", "past_key_0", "past_key_1"]
    assert [o.name for o in model.graph.output][:2] == ["logits", "present_key_0"]
    manifest = json.load(open(result.manifest_path))
    assert manifest["cache_inplace"] and manifest["cache_layout"] == "BHSD"
    assert manifest["num_layers"] == 2 and manifest["max_input_len"] == 8
    assert manifest["caches"][0]["key"]["shape"] == [1, 2, MAX_CACHE, 16]


@caps.requires_onnx
def test_onnx_export_rejects_mutable_cache(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.onnx import ONNXExportConfig

    cfg = ONNXExportConfig(output_path=str(tmp_path / "m.onnx"), backend="onnx",
                           convert_options={**CONVERT, "cache_impl": "mutable"})
    with pytest.raises(ValueError, match="mutable"):
        export_to_backend(_tiny(), config=cfg)


@caps.requires_onnx
@pytest.mark.parametrize("precision", [4, 8])
def test_weight_quantization_pass_is_exact(precision):
    """Per-channel scales survive the Q/DQ rewrite exactly (INT4 tiles them over blocks)."""
    from onnx import TensorProto, helper, numpy_helper

    from transformersurgeon.export.onnx import apply_weight_quantization

    rng = np.random.default_rng(0)
    qmax = 7 if precision == 4 else 127
    scale = rng.uniform(0.01, 0.1, size=64).astype(np.float32)
    q = rng.integers(-qmax - (precision == 4), qmax + 1, size=(64, 256))
    w = (q * scale[:, None]).astype(np.float16)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "lin.weight"], ["y"], transB=1)], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 256])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 64])],
        [numpy_helper.from_array(w, "lin.weight")])
    model = helper.make_model(graph)
    info = {"lin": {"precision": precision, "scale": torch.tensor(scale), "per_channel": True, "act_scale": None}}
    done = apply_weight_quantization(model, info)
    assert done == {"lin": {"precision": precision, "granularity": "per_channel"}}

    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    q_out, s_out = inits[dq.input[0]].astype(np.float32), inits[dq.input[1]].astype(np.float32)
    if precision == 4:
        block = next(a.i for a in dq.attribute if a.name == "block_size")
        s_out = np.repeat(s_out, block, axis=1)
    else:
        s_out = s_out[:, None]
    np.testing.assert_array_equal(q_out, q)
    np.testing.assert_allclose(q_out * s_out, w.astype(np.float32), rtol=1e-3)


@caps.requires_tensorrt_py
def test_tensorrt_session_matches_eager(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.tensorrt import TensorRTONNXExportConfig
    from transformersurgeon.export.tensorrt.session import TensorRTLLMSession

    model = _tiny(torch.float16)
    cfg = TensorRTONNXExportConfig(output_path=str(tmp_path / "model.onnx"), backend="tensorrt_onnx",
                                   max_input_len=8, convert_options=CONVERT)
    result = export_to_backend(model, config=cfg)
    session = TensorRTLLMSession(result.manifest_path)
    assert session.decoder.aliased_input("present_key_0") == "past_key_0"  # in-place KV

    prompt = torch.tensor([3, 17, 42, 5, 99])
    session.prefill(prompt)
    ref = _eager_logits(model, prompt, "cuda")
    torch.testing.assert_close(session.logits.float(), ref, atol=2e-2, rtol=2e-2)

    # Greedy decode through the CUDA-graph step must match eager over the grown prompt.
    tokens = prompt.tolist() + [int(session.input_ids.item())]
    for _ in range(3):
        session.step()
        ref = _eager_logits(model, torch.tensor(tokens), "cuda")
        torch.testing.assert_close(session.logits.float(), ref, atol=2e-2, rtol=2e-2)
        tokens.append(int(session.input_ids.item()))


@caps.requires_edgellm_plugin
def test_edgellm_int4_plugin_matches_dequantize_backend(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.tensorrt import TensorRTONNXExportConfig
    from transformersurgeon.export.tensorrt.session import TensorRTLLMSession
    from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager

    prompt = torch.tensor([3, 17, 42, 5, 99])
    logits = {}
    for backend in ("dequantize", "edgellm_plugin"):
        model = _tiny(torch.float16)
        manager = Qwen2CompressionSchemesManager(model)
        crit = ["self_attn", "mlp"]
        manager.set("quantization", "precision", 4, criteria=crit)
        manager.set("quantization", "granularity", "per_channel", criteria=crit)
        manager.apply(hard=True, criteria=crit)
        cfg = TensorRTONNXExportConfig(output_path=str(tmp_path / backend / "model.onnx"), backend="tensorrt_onnx",
                                       max_input_len=8, convert_options=CONVERT, int4_backend=backend)
        result = export_to_backend(model, config=cfg)
        manifest = json.load(open(result.manifest_path))
        if backend == "edgellm_plugin":
            plugin = manifest["plugins"]["edgellm"]
            # 64/128-wide projections take the plugin; the 32-wide k/v fall back to Q/DQ.
            assert plugin["layers"] > 0 and plugin["fallback_layers"] > 0
        session = TensorRTLLMSession(result.manifest_path)
        session.prefill(prompt)
        logits[backend] = session.logits.float().clone()
    torch.testing.assert_close(logits["edgellm_plugin"], logits["dequantize"], atol=3e-2, rtol=3e-2)
