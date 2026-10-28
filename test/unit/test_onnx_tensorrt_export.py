"""ONNX export ("onnx") and plain-TensorRT ("tensorrt") backends on a tiny
random-weight Qwen2 -- no downloads. TensorRT tests are capability-gated.

See docs/investigations/TENSORRT_ONNX_EXPORT.md for why this path exists.
"""
from __future__ import annotations

import json
import os

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
    from transformersurgeon.export.convert import convert_for_export

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


def _act_info(scale=0.05, scheme="symmetric", zero_point=0, out_scale=None):
    return {"act_scale": torch.tensor([scale]), "act_zero_point": torch.tensor([zero_point], dtype=torch.int32),
            "act_precision": 8, "act_scheme": scheme, "act_out_scale": out_scale}


def _gemm_model(weight_names=("lin.weight",)):
    from onnx import TensorProto, helper, numpy_helper

    nodes = [helper.make_node("Gemm", ["x", w], [f"y{i}"], transB=1) for i, w in enumerate(weight_names)]
    inits = [numpy_helper.from_array(np.ones((4, 8), np.float16), w) for w in weight_names]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8])],
                              [helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT16, [1, 4])
                               for i in range(len(weight_names))], inits)
    return helper.make_model(graph)


@caps.requires_onnx
def test_activation_quantization_pass_places_qdq_on_gemm_input():
    from onnx import numpy_helper

    from transformersurgeon.export.onnx import apply_activation_quantization

    model = _gemm_model()
    done = apply_activation_quantization(model, {"lin": _act_info(0.05)})
    assert done == {"lin": {"act_precision": 8, "act_granularity": "per_tensor"}}
    ops = [n.op_type for n in model.graph.node]
    assert ops == ["QuantizeLinear", "DequantizeLinear", "Gemm"]
    gemm = model.graph.node[2]
    assert gemm.input[0] == model.graph.node[1].output[0] and model.graph.node[0].input[0] == "x"
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    assert inits["lin.act_scale"].dtype == np.float16 and float(inits["lin.act_scale"]) == pytest.approx(0.05, rel=1e-3)
    assert inits["lin.act_zero_point"].dtype == np.int8 and int(inits["lin.act_zero_point"]) == 0


@caps.requires_onnx
@pytest.mark.parametrize("bad", [dict(scheme="asymmetric", zero_point=3), dict(out_scale=torch.tensor([0.1]))])
def test_activation_quantization_pass_rejects_non_tensorrt_schemes(bad):
    from transformersurgeon.export.onnx import apply_activation_quantization

    with pytest.raises(NotImplementedError):
        apply_activation_quantization(_gemm_model(), {"lin": _act_info(**bad)})


@caps.requires_tensorrt
@pytest.mark.parametrize("smooth", [False, True])
def test_tensorrt_w8a8_static_activations_match_eager(tmp_path, smooth):
    """W8 weights + calibrated static INT8 input activations (TensorRT W8A8), optionally
    SmoothQuant-ed first, == the hook-simulated tsurgeon model; the ONNX carries one
    activation Q/DQ per linear."""
    import onnx
    from torch.utils.data import DataLoader

    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.tensorrt import TensorRTExportConfig
    from transformersurgeon.export.tensorrt.session import TensorRTLLMSession
    from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager

    model = _tiny(torch.float16)
    manager = Qwen2CompressionSchemesManager(model)
    crit = ["self_attn", "mlp"]
    for key, value in (("precision", 8), ("granularity", "per_channel"), ("precision_activation", 8),
                       ("method_activation", "maxmin"), ("scheme_activation", "symmetric"),
                       ("quantize_output_activation", False)):
        manager.set("quantization", key, value, criteria=crit)
    gen = torch.Generator().manual_seed(1)
    manager.set_calibration_data(DataLoader(
        [{"input_ids": torch.randint(0, 256, (16,), generator=gen)} for _ in range(4)], batch_size=1))
    if smooth:
        assert manager.smooth_activations(alpha=0.5, criteria=crit)["groups"] == 6
    manager.apply(hard=True, criteria=crit, show_progress=False)
    linears = [m for m in model.modules() if hasattr(m, "_act_quant_scale")]
    assert len(linears) == 14 and not any(hasattr(m, "_act_out_quant_scale") for m in linears)

    prompt = torch.tensor([3, 17, 42, 5, 99])
    with torch.no_grad():
        ref = model.cuda()(input_ids=prompt[None].cuda()).logits[0, -1].float()
    model.cpu()
    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), backend="tensorrt",
                               max_input_len=8, convert_options=CONVERT)
    result = export_to_backend(model, config=cfg)
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    assert sum(n.op_type == "QuantizeLinear" for n in graph.node) == 14
    manifest = json.load(open(result.manifest_path))
    assert all(q.get("act_precision") == 8 for q in manifest["quantized_layers"].values())

    session = TensorRTLLMSession(result.manifest_path)
    session.prefill(prompt)
    torch.testing.assert_close(session.logits.float(), ref, atol=3e-2, rtol=3e-2)


@caps.requires_tensorrt
def test_tensorrt_session_matches_eager(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.tensorrt import TensorRTExportConfig
    from transformersurgeon.export.tensorrt.session import TensorRTLLMSession

    model = _tiny(torch.float16)
    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), backend="tensorrt",
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


@caps.requires_tensorrt
def test_engine_file_is_streamed_in_chunks(tmp_path, monkeypatch):
    """An engine file is deserialized through the chunked stream reader (never
    read whole), and gives the same logits as the in-memory plan."""
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.tensorrt import TensorRTExportConfig, engine as engine_mod
    from transformersurgeon.export.tensorrt.session import TensorRTLLMSession

    model = _tiny(torch.float16)
    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), backend="tensorrt",
                               max_input_len=8, convert_options=CONVERT)
    result = export_to_backend(model, config=cfg)
    reads = []
    make_reader = engine_mod._file_stream_reader

    def small_chunks(path, chunk_bytes=0):
        r = make_reader(path, chunk_bytes=4096)
        read = r.read
        r.read = lambda n, s: reads.append(n) or read(n, s)
        return r

    monkeypatch.setattr(engine_mod, "_file_stream_reader", small_chunks)
    session = TensorRTLLMSession(result.manifest_path)
    assert len(reads) > 1
    prompt = torch.tensor([3, 17, 42, 5, 99])
    session.prefill(prompt)
    streamed = session.logits.float().clone()

    with open(os.path.join(os.path.dirname(result.manifest_path), "model.engine"), "rb") as f:
        runner = engine_mod.TensorRTEngineRunner(f.read())  # in-memory plan path
    assert runner.engine.num_io_tensors == session.decoder.engine.num_io_tensors
    torch.testing.assert_close(streamed, _eager_logits(model, prompt, "cuda"), atol=2e-2, rtol=2e-2)


@caps.requires_edgellm_plugin
def test_edgellm_int4_plugin_matches_dequantize_backend(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.tensorrt import TensorRTExportConfig
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
        cfg = TensorRTExportConfig(output_path=str(tmp_path / backend / "model.onnx"), backend="tensorrt",
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
