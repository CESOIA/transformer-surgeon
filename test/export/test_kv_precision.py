"""KV-cache precision through the export pipeline: options, conversion, wrapper, manifest, backends.

The model is a tiny random Qwen2, or with ``pytest --kv-model NAME`` a pretrained one (e.g.
Qwen/Qwen2.5-0.5B): every test then runs on it, with the accuracy bounds of a deep model.
"""
import gc
import json
import os
import re
import shutil

import pytest
import torch

from _helpers import capabilities as caps
from _helpers import model_factory
from transformersurgeon.blocks import KV_PRECISIONS, kv_precision_pair
from transformersurgeon.export import (
    LLMWrapper, build_llm_manifest, build_zero_caches, convert_for_export, write_cache_rows,
    zero_caches_from_manifest,
)
from transformersurgeon.export.operations import check_components, options_for, resolve_convert_options
from transformersurgeon.utils.devices import move_to_device

CACHE = 16
PRETRAINED = model_factory.KV_MODEL is not None
# A pretrained model is far deeper than the tiny one and compounds every rounding difference over its layers,
# so its checks are relative logit errors (|got - expected| / |expected|) over the tests' few tokens, with margin
# over Qwen2.5-0.5B's. A backend against the eager model with the same cache, where a value on a rounding boundary
# can round the other way (int4 and nvfp4 codes are coarse): fp32 ONNX Runtime (Qwen2.5-0.5B: 1e-5, 7e-6, 0.02,
# 2e-5) and fp16 TensorRT (0.01, 0.05, 0.26, 0.05, 0.25).
ONNX_ERROR = {"full": 1e-3, "int8": 1e-3, "int4": 0.06, "fp8": 1e-3}
ENGINE_ERROR = {"full": 0.05, "int8": 0.15, "int4": 0.5, "fp8": 0.15, "nvfp4": 0.5}
# A quantised cache against the float cache (Qwen2.5-0.5B: int8 0.22, int4 0.69; the per-row scale follows
# the few large key channels, so the small ones get few codes).
CACHE_ERROR = {"int8": 0.4, "int4": 1.0} if PRETRAINED else {"int8": 0.02, "int4": 0.25}


def _model(dtype=torch.float32):
    return model_factory.kv_model(dtype)


def _geometry(model):
    """``(layers, kv_heads, head_dim)`` of the model's caches."""
    config = model.config
    return config.num_hidden_layers, config.num_key_value_heads, config.hidden_size // config.num_attention_heads


def _relative_error(got, expected):
    return ((got - expected).norm() / expected.norm()).item()


def _assert_backend_matches(got, expected, atol, pretrained_error):
    """A backend's logits reproduce the eager model's with the same cache: each within ``atol`` on the tiny
    model, within a relative error of ``pretrained_error`` on a pretrained one."""
    if PRETRAINED:
        assert _relative_error(got, expected) < pretrained_error
    else:
        torch.testing.assert_close(got, expected, atol=atol, rtol=atol)


def _wrapper(model, cache_impl="mutable", kv_precision="full"):
    options = {"cache_impl": cache_impl, "max_cache_len": CACHE, "kv_precision": kv_precision}
    decoder = convert_for_export(model, options=options)["text"]
    return LLMWrapper(model.get_input_embeddings(), decoder, model.lm_head).eval()


def _greedy_logits(wrapper, tokens):
    """Logits of every step of a token-by-token run, with the wrapper's own caches (io_rows: the
    caller writes the rows each step returns)."""
    keys, values = build_zero_caches(wrapper.decoder)
    out = []
    with torch.no_grad():
        for t, token in enumerate(tokens):
            args = (torch.tensor([token]), torch.tensor([t]))
            if wrapper.cache_impl == "io_rows":
                logits, new_keys, new_values = wrapper(*args, keys, values)
                for cache, rows in zip(keys + values, new_keys + new_values):
                    write_cache_rows(cache, rows, args[1])
            elif wrapper.graph_io_cache:
                logits, keys, values = wrapper(*args, keys, values)
            else:
                logits = wrapper(*args)
            out.append(logits)
    return torch.stack(out)


@pytest.mark.parametrize("backend, supported", [
    ("xnnpack", {"full", "int8", "int4"}), ("onnx", {"full", "int8", "int4", "fp8"}),
    ("qnn", {"full"}), ("tensorrt", {"full", "int8", "int4", "fp8", "nvfp4"}),
])
def test_backends_list_the_precisions_they_support(backend, supported):
    assert set(options_for(backend, "kv_precision")) == supported
    assert resolve_convert_options(backend)["kv_precision"] == "full"
    for name in supported:
        assert resolve_convert_options(backend, {"kv_precision": name})["kv_precision"] == name
    if "int8" not in supported:
        with pytest.raises(ValueError, match=f"not supported by the {backend} backend"):
            resolve_convert_options(backend, {"kv_precision": "int8"})


def test_a_precision_selects_the_cache_it_needs():
    """fp8 and nvfp4 rows need io_rows everywhere; on TensorRT the integer rows need it too (uint8 only at
    the graph boundary). Without an explicit cache the resolution picks it; a conflicting cache is rejected."""
    assert resolve_convert_options("onnx", {"kv_precision": "int8"})["cache_impl"] == "io_inplace"
    assert resolve_convert_options("onnx", {"kv_precision": "fp8"})["cache_impl"] == "io_rows"
    with pytest.raises(ValueError, match="kv_precision implementation 'nvfp4' is not supported by the onnx"):
        resolve_convert_options("onnx", {"kv_precision": "nvfp4"})
    for name in ("int8", "int4", "fp8", "nvfp4"):
        assert resolve_convert_options("tensorrt", {"kv_precision": name})["cache_impl"] == "io_rows"
        with pytest.raises(ValueError, match="needs cache_impl='io_rows' on the tensorrt backend"):
            resolve_convert_options("tensorrt", {"kv_precision": name, "cache_impl": "io_inplace"})
    with pytest.raises(ValueError, match="kv_precision implementation 'fp8' is not supported by the xnnpack"):
        resolve_convert_options("xnnpack", {"kv_precision": "fp8"})
    decoder = convert_for_export(_model(), options={"cache_impl": "io_inplace", "max_cache_len": CACHE,
                                                    "kv_precision": "int8"})["text"]
    check_components("onnx", decoder)
    with pytest.raises(ValueError, match="needs cache_impl='io_rows'"):
        check_components("tensorrt", decoder)


def test_conversion_builds_integer_caches_and_backends_check_them():
    decoder = convert_for_export(_model(), options={"max_cache_len": CACHE, "kv_precision": "int4"})["text"]
    assert all(b.attn.kv_precision == "int4" and b.attn.key_cache.dtype == torch.uint8 for b in decoder.blocks)
    check_components("xnnpack", decoder)
    with pytest.raises(ValueError, match="kv_precision implementation 'int4' is not supported by the qnn"):
        check_components("qnn", decoder)


def test_a_precision_pair_is_resolved_and_checked_per_cache():
    """kv_precision=(key, value): each part must be an option of the backend, the pair selects the cache either part
    needs, and a pair of equal parts is the single name."""
    resolved = resolve_convert_options("onnx", {"kv_precision": ("int8", "int4")})
    assert (resolved["kv_precision"], resolved["cache_impl"]) == (("int8", "int4"), "io_inplace")
    assert resolve_convert_options("onnx", {"kv_precision": ("int8", "fp8")})["cache_impl"] == "io_rows"
    assert resolve_convert_options("tensorrt", {"kv_precision": ("full", "nvfp4")})["cache_impl"] == "io_rows"
    assert resolve_convert_options("xnnpack", {"kv_precision": ["int8", "int8"]})["kv_precision"] == "int8"
    assert resolve_convert_options("qnn", {"kv_precision": ("full", "full")})["kv_precision"] == "full"
    with pytest.raises(ValueError, match="kv_precision implementation 'fp8' is not supported by the xnnpack"):
        resolve_convert_options("xnnpack", {"kv_precision": ("int8", "fp8")})
    with pytest.raises(ValueError, match="kv_precision implementation 'int8' is not supported by the qnn"):
        resolve_convert_options("qnn", {"kv_precision": ("full", "int8")})
    with pytest.raises(ValueError, match="needs cache_impl='io_rows' on the tensorrt backend"):
        resolve_convert_options("tensorrt", {"kv_precision": ("int8", "full"), "cache_impl": "io_inplace"})
    for pair in (("int8", "nvfp4"), ("nvfp4", "int4")):   # TensorRT 11.4 crashes building such an engine
        with pytest.raises(ValueError, match="cannot be in one tensorrt graph"):
            resolve_convert_options("tensorrt", {"kv_precision": pair})
    decoder = convert_for_export(_model(), options={"cache_impl": "io_inplace", "max_cache_len": CACHE,
                                                    "kv_precision": ("int8", "int4")})["text"]
    assert all(b.attn.kv_precision == ("int8", "int4") for b in decoder.blocks)
    check_components("onnx", decoder)
    with pytest.raises(ValueError, match="needs cache_impl='io_rows'"):
        check_components("tensorrt", decoder)


def _signature(cache):
    """Shapes and dtypes of a cache, or of each part of a ``(rows, scales)`` pair."""
    return [(t.shape, t.dtype) for t in cache] if isinstance(cache, (tuple, list)) else (cache.shape, cache.dtype)


def test_manifest_of_a_precision_pair_gives_each_cache_its_dtype():
    model = _model()
    wrapper = _wrapper(model, "io_inplace", ("int8", "full"))
    manifest = json.loads(json.dumps(build_llm_manifest(wrapper, model.config, backend="onnx", artifact="m.onnx")))
    _, kv_heads, head_dim = _geometry(model)
    layer = manifest["caches"][0]
    assert manifest["kv_precision"] == ["int8", "full"]
    assert (layer["key"]["dtype"], layer["key"]["shape"]) == ("uint8", [1, kv_heads, CACHE, head_dim + 2])
    assert (layer["value"]["dtype"], layer["value"]["shape"]) == ("float32", [1, kv_heads, CACHE, head_dim])
    for got, expected in zip(zero_caches_from_manifest(manifest), build_zero_caches(wrapper.decoder)):
        assert [_signature(c) for c in got] == [_signature(c) for c in expected]


@caps.requires_float4
def test_only_the_nvfp4_cache_of_a_pair_gets_block_scales():
    """The manifest, the graph I/O names and the zero caches give block scales to the cache stored as nvfp4
    only: the value cache is a (rows, scales) pair, the key cache one tensor."""
    from transformersurgeon.export.manifest import cache_scales, graph_io_names

    model = _model()
    layers = model.config.num_hidden_layers
    wrapper = _wrapper(model, "io_rows", ("fp8", "nvfp4"))
    assert cache_scales(wrapper.decoder) == ("value",)
    manifest = json.loads(json.dumps(build_llm_manifest(wrapper, model.config, backend="tensorrt", artifact="m.onnx")))
    layer = manifest["caches"][0]
    assert "value_scale" in layer and "key_scale" not in layer
    assert (layer["key"]["dtype"], layer["value"]["dtype"]) == ("float8_e4m3fn", "float4_e2m1fn_x2")
    inputs, outputs = graph_io_names(layers, scales=("value",))
    assert inputs[2:] == [f"past_key_{i}" for i in range(layers)] + \
        [name for i in range(layers) for name in (f"past_value_{i}", f"past_value_scale_{i}")]
    assert outputs[1:] == [name.replace("past_", "present_") for name in inputs[2:]]
    for got, expected in zip(zero_caches_from_manifest(manifest), build_zero_caches(wrapper.decoder)):
        assert [_signature(c) for c in got] == [_signature(c) for c in expected]


@pytest.mark.parametrize("cache_impl", ["mutable", "io_inplace"])
@pytest.mark.parametrize("name", ["int8", "int4"])
def test_integer_cache_wrapper_tracks_the_float_wrapper(cache_impl, name):
    model = _model()
    tokens = [3, 17, 42, 5, 99, 7]
    expected = _greedy_logits(_wrapper(model, cache_impl), tokens)
    got = _greedy_logits(_wrapper(model, cache_impl, name), tokens)
    assert _relative_error(got, expected) < CACHE_ERROR[name]


def test_manifest_records_the_precision_and_gives_uint8_caches(tmp_path):
    model = _model()
    wrapper = _wrapper(model, "io_inplace", "int8")
    manifest = build_llm_manifest(wrapper, model.config, backend="onnx", artifact=str(tmp_path / "model.onnx"))
    assert manifest["kv_precision"] == "int8" and manifest["cache_dtype"] == "uint8"
    _, kv_heads, head_dim = _geometry(model)
    assert manifest["caches"][0]["key"]["shape"] == [1, kv_heads, CACHE, head_dim + 2]
    manifest = json.loads(json.dumps(manifest))
    for got, expected in zip(zero_caches_from_manifest(manifest), build_zero_caches(wrapper.decoder)):
        assert [(t.shape, t.dtype) for t in got] == [(t.shape, t.dtype) for t in expected]
    assert build_llm_manifest(_wrapper(model), model.config, backend="xnnpack", artifact="m.pte")["kv_precision"] \
        == "full"


def test_manifest_of_an_fp8_rows_cache():
    model = _model()
    wrapper = _wrapper(model, "io_rows", "fp8")
    manifest = json.loads(json.dumps(build_llm_manifest(wrapper, model.config, backend="tensorrt", artifact="m.onnx")))
    assert (manifest["kv_precision"], manifest["cache_dtype"], manifest["cache_layout"]) == \
        ("fp8", "float8_e4m3fn", "BHSD")
    assert manifest["cache_rows"] and not manifest["cache_inplace"]
    _, kv_heads, head_dim = _geometry(model)
    assert manifest["caches"][0]["key"]["shape"] == [1, kv_heads, CACHE, head_dim]
    for got, expected in zip(zero_caches_from_manifest(manifest), build_zero_caches(wrapper.decoder)):
        assert [(t.shape, t.dtype) for t in got] == [(t.shape, t.dtype) for t in expected]


@caps.requires_float4
def test_manifest_of_an_nvfp4_rows_cache():
    """An NVFP4 cache is two tensors: FP4 rows and FP8 block scales, each with its own manifest entry
    (shape, dtype, I/O names), and zero caches from the manifest are the same (rows, scales) pairs."""
    model = _model()
    wrapper = _wrapper(model, "io_rows", "nvfp4")
    manifest = json.loads(json.dumps(build_llm_manifest(wrapper, model.config, backend="tensorrt", artifact="m.onnx")))
    assert (manifest["kv_precision"], manifest["cache_dtype"], manifest["cache_rows"]) == \
        ("nvfp4", "float4_e2m1fn_x2", True)
    _, kv_heads, head_dim = _geometry(model)   # a multiple of 16: whole blocks per row
    layer = manifest["caches"][1]
    assert layer["key"] == {"input": "past_key_1", "output": "present_key_1",
                            "shape": [1, kv_heads, CACHE, head_dim // 2], "dtype": "float4_e2m1fn_x2"}
    assert layer["value_scale"] == {"input": "past_value_scale_1", "output": "present_value_scale_1",
                                    "shape": [1, kv_heads, CACHE, head_dim // 16], "dtype": "float8_e4m3fn"}
    for got, expected in zip(zero_caches_from_manifest(manifest), build_zero_caches(wrapper.decoder)):
        assert [[(t.shape, t.dtype) for t in pair] for pair in got] == \
            [[(t.shape, t.dtype) for t in pair] for pair in expected]


@caps.requires_onnx
@caps.requires_float4
def test_onnx_graph_of_an_nvfp4_cache_has_fp4_rows_and_fp8_scales(tmp_path):
    """The exported graph (TensorRT backend, no engine) takes and returns FLOAT4E2M1 rows next to
    FLOAT8E4M3FN block scales, in the order the manifest names them, and quantizes with a block size
    of 16."""
    import onnx

    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import TensorRTExportConfig
    from transformersurgeon.export.manifest import graph_io_names

    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), max_input_len=4, build_engine=False,
                               convert_options={"max_cache_len": CACHE, "kv_precision": "nvfp4"})
    model = _model(torch.float16)
    layers, kv_heads, head_dim = _geometry(model)
    result = export_to_backend(model, cfg)
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    inputs, outputs = graph_io_names(layers, scales=True)
    assert [i.name for i in graph.input] == inputs and [o.name for o in graph.output] == outputs
    types = {i.name: (i.type.tensor_type.elem_type, [d.dim_value for d in i.type.tensor_type.shape.dim])
             for i in graph.input}
    assert types["past_key_0"] == (onnx.TensorProto.FLOAT4E2M1, [1, kv_heads, CACHE, head_dim])
    assert types["past_value_scale_1"] == (onnx.TensorProto.FLOAT8E4M3FN, [1, kv_heads, CACHE, head_dim // 16])
    fp4_quantize = [n for n in graph.node if n.op_type == "QuantizeLinear"
                    and onnx.helper.get_attribute_value(next(a for a in n.attribute if a.name == "output_dtype"))
                    == onnx.TensorProto.FLOAT4E2M1]
    assert len(fp4_quantize) == 2 * layers   # keys and values of every layer
    assert all(next(a.i for a in n.attribute if a.name == "block_size") == 16 for n in fp4_quantize)
    assert json.load(open(result.manifest_path))["cache_impl"] == "io_rows"


@caps.requires_onnx
@caps.requires_float4
def test_onnx_graph_of_a_pair_with_an_nvfp4_value_cache(tmp_path):
    """kv_precision=("fp8", "nvfp4") (TensorRT backend, no engine): FP8 key rows, FP4 value rows next to FP8
    block scales, in the order the manifest names them."""
    import onnx

    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import TensorRTExportConfig
    from transformersurgeon.export.manifest import graph_io_names

    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), max_input_len=4, build_engine=False,
                               convert_options={"max_cache_len": CACHE, "kv_precision": ("fp8", "nvfp4")})
    model = _model(torch.float16)
    layers, kv_heads, head_dim = _geometry(model)
    result = export_to_backend(model, cfg)
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    inputs, outputs = graph_io_names(layers, scales=("value",))
    assert [i.name for i in graph.input] == inputs and [o.name for o in graph.output] == outputs
    types = {i.name: (i.type.tensor_type.elem_type, [d.dim_value for d in i.type.tensor_type.shape.dim])
             for i in graph.input}
    assert types["past_key_0"] == (onnx.TensorProto.FLOAT8E4M3FN, [1, kv_heads, CACHE, head_dim])
    assert types["past_value_0"] == (onnx.TensorProto.FLOAT4E2M1, [1, kv_heads, CACHE, head_dim])
    assert types["past_value_scale_1"] == (onnx.TensorProto.FLOAT8E4M3FN, [1, kv_heads, CACHE, head_dim // 16])
    assert json.load(open(result.manifest_path))["kv_precision"] == ["fp8", "nvfp4"]


@caps.requires_executorch
def test_xnnpack_program_stores_keys_and_values_with_their_own_precision(tmp_path):
    """kv_precision=("int8", "int4") on XNNPACK: the program's mutable caches are stored differently and, run in
    ExecuTorch's runtime token by token, the program reproduces the eager model with the same pair."""
    from executorch.runtime import Runtime

    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import XNNPACKExportConfig

    pair, tokens = ("int8", "int4"), [3, 17, 42, 5, 99, 7]
    model = _model()
    _, _, head_dim = _geometry(model)
    pte = str(tmp_path / "model.pte")
    cfg = XNNPACKExportConfig(output_path=pte, max_seq_len=CACHE, run_weight_mismatch_check=False,
                              convert_options={"max_cache_len": CACHE, "kv_precision": pair})
    manifest = json.load(open(export_to_backend(_model(), cfg).manifest_path))
    assert manifest["kv_precision"] == list(pair)
    layer = manifest["caches"][0]
    assert (layer["key"]["shape"][-1], layer["value"]["shape"][-1]) == (head_dim + 2, head_dim // 2 + 2)

    method = Runtime.get().load_program(pte).load_method("forward")
    got = torch.stack([method.execute([torch.tensor([token]), torch.tensor([t])])[0].reshape(-1)
                       for t, token in enumerate(tokens)])
    expected = _greedy_logits(_wrapper(model, "mutable", pair), tokens)
    _assert_backend_matches(got, expected.reshape(got.shape), 1e-3, ONNX_ERROR["int4"])


@caps.requires_executorch
@pytest.mark.parametrize("name", ["int8", "int4"])
def test_xnnpack_export_with_an_integer_cache_writes_a_program(tmp_path, name):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import XNNPACKExportConfig

    pte = str(tmp_path / "model.pte")
    cfg = XNNPACKExportConfig(output_path=pte, max_seq_len=CACHE, run_weight_mismatch_check=False,
                              convert_options={"max_cache_len": CACHE, "kv_precision": name})
    result = export_to_backend(_model(), cfg)
    assert os.path.getsize(pte) > 0
    manifest = json.load(open(result.manifest_path))
    assert manifest["kv_precision"] == name and manifest["cache_dtype"] == "uint8"


@caps.requires_onnx
@pytest.mark.parametrize("name", ["int8", "int4"])
def test_onnx_export_with_an_integer_cache_has_uint8_cache_io(tmp_path, name):
    import onnx

    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import ONNXExportConfig

    cfg = ONNXExportConfig(output_path=str(tmp_path / "model.onnx"), max_input_len=4,
                           convert_options={"max_cache_len": CACHE, "kv_precision": name})
    result = export_to_backend(_model(), cfg)
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    caches = [i for i in graph.input if i.name.startswith("past_")]
    assert caches and all(i.type.tensor_type.elem_type == onnx.TensorProto.UINT8 for i in caches)
    assert json.load(open(result.manifest_path))["kv_precision"] == name

    ort = pytest.importorskip("onnxruntime")
    try:
        session = ort.InferenceSession(result.onnx_path)
    except Exception as exc:  # an onnxruntime without the opset-24 TensorScatter
        pytest.skip(f"onnxruntime cannot load the exported model: {exc}")
    wrapper = _wrapper(_model(), "io_inplace", name)
    keys, values = build_zero_caches(wrapper.decoder)
    feeds = {"input_ids": torch.tensor([3]).numpy(), "pos_id": torch.tensor([0]).numpy()}
    feeds.update({f"past_key_{i}": k.numpy() for i, k in enumerate(keys)})
    feeds.update({f"past_value_{i}": v.numpy() for i, v in enumerate(values)})
    logits = session.run(["logits"], feeds)[0]
    with torch.no_grad():
        expected = wrapper(torch.tensor([3]), torch.tensor([0]), keys, values)[0]
    _assert_backend_matches(torch.from_numpy(logits).reshape(expected.shape), expected, 1e-3, ONNX_ERROR[name])


def _run_onnx(session, manifest, tokens):
    """Decode ``tokens`` one by one in ONNX Runtime: the model's own caches are fed back, or (io_rows)
    the rows it returns are written into the caches."""
    import onnx
    import onnxruntime as ort

    def ort_value(t):
        if t.dtype == torch.float8_e4m3fn:  # numpy has no FP8: hand over the bytes with their ONNX type
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(t.view(torch.uint8).numpy(),
                                                                     onnx.TensorProto.FLOAT8E4M3FN)
        return ort.OrtValue.ortvalue_from_numpy(t.numpy())

    keys, values = zero_caches_from_manifest(manifest)
    caches = {**{f"past_key_{i}": k for i, k in enumerate(keys)}, **{f"past_value_{i}": v for i, v in enumerate(values)}}
    outputs = [o.name for o in session.get_outputs()]
    steps = []
    for t, token in enumerate(tokens):
        feeds = {"input_ids": torch.tensor([token]), "pos_id": torch.tensor([t]), **caches}
        results = session.run_with_ort_values(outputs, {name: ort_value(v) for name, v in feeds.items()})
        results = {name: torch.from_numpy(r.numpy()) for name, r in zip(outputs, results)}
        for name, value in results.items():
            if name != "logits":
                past = name.replace("present_", "past_")
                value = value.view(caches[past].dtype)  # FP8 comes back as its bytes
                if manifest["cache_rows"]:
                    write_cache_rows(caches[past], value, torch.tensor([t]))
                else:
                    caches[past] = value
        steps.append(results["logits"].reshape(-1))
    return torch.stack(steps), caches


@caps.requires_onnx
def test_chosen_kv_precision_reaches_the_exported_model(tmp_path):
    """End to end: choose the KV precision, export, and show the exported model really uses it.

    For each choice the ONNX graph's cache inputs have that precision's stored dtype and row width,
    the manifest records it, and ONNX Runtime decoding several tokens with its own caches reproduces
    the eager model built with the same choice (fp8 selects cache_impl="io_rows": the rows the graph
    returns are written by the caller). The quantised choices change the result (they are not
    ignored) and int4 moves it further from the float cache than int8 or fp8.
    """
    import onnx
    ort = pytest.importorskip("onnxruntime")
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import ONNXExportConfig

    model = _model()
    _, _, head_dim = _geometry(model)
    tokens = [3, 17, 42, 5, 99, 7]
    logits = {}
    stored_types = {"full": onnx.TensorProto.FLOAT, "int8": onnx.TensorProto.UINT8, "int4": onnx.TensorProto.UINT8,
                    "fp8": onnx.TensorProto.FLOAT8E4M3FN}
    for name in ("full", "int8", "int4", "fp8"):
        cfg = ONNXExportConfig(output_path=str(tmp_path / name / "model.onnx"),
                               convert_options={"max_cache_len": CACHE, "kv_precision": name})
        result = export_to_backend(_model(), cfg)

        manifest = json.load(open(result.manifest_path))
        assert manifest["kv_precision"] == name
        assert manifest["cache_impl"] == ("io_rows" if name == "fp8" else "io_inplace")
        cache_inputs = [i for i in onnx.load(result.onnx_path, load_external_data=False).graph.input
                        if i.name.startswith("past_")]
        assert all(i.type.tensor_type.elem_type == stored_types[name] for i in cache_inputs)
        assert cache_inputs[0].type.tensor_type.shape.dim[-1].dim_value == KV_PRECISIONS[name].row_width(head_dim)

        try:
            session = ort.InferenceSession(result.onnx_path)
        except Exception as exc:  # an onnxruntime without the opset-24 TensorScatter
            pytest.skip(f"onnxruntime cannot load the exported model: {exc}")
        logits[name], caches = _run_onnx(session, manifest, tokens)
        assert {c.dtype for c in caches.values()} == {KV_PRECISIONS[name].storage_dtype(torch.float32)}
        eager = _greedy_logits(_wrapper(model, manifest["cache_impl"], name), tokens)
        _assert_backend_matches(logits[name], eager, 1e-3, ONNX_ERROR[name])
        del session
        shutil.rmtree(tmp_path / name)   # a pretrained model's export takes gigabytes

    def distance(name):
        return _relative_error(logits[name], logits["full"])

    assert 0 < distance("int8") < distance("int4") < CACHE_ERROR["int4"]
    assert 0 < distance("fp8") < distance("int4")


@caps.requires_onnx
@pytest.mark.parametrize("pair", [("int8", "int4"), ("full", "int8"), ("fp8", "int4")])
def test_onnx_model_stores_keys_and_values_with_their_own_precision(tmp_path, pair):
    """kv_precision=(key, value) end to end on ONNX: the graph's key and value cache inputs have their own stored
    dtype and row width, the manifest records the pair and each cache's dtype, and ONNX Runtime decoding several
    tokens with its own caches reproduces the eager model built with the same pair (fp8 selects io_rows)."""
    import onnx
    ort = pytest.importorskip("onnxruntime")
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import ONNXExportConfig

    model = _model()
    _, _, head_dim = _geometry(model)
    stored_types = {"full": onnx.TensorProto.FLOAT, "int8": onnx.TensorProto.UINT8, "int4": onnx.TensorProto.UINT8,
                    "fp8": onnx.TensorProto.FLOAT8E4M3FN}
    cfg = ONNXExportConfig(output_path=str(tmp_path / "model.onnx"),
                           convert_options={"max_cache_len": CACHE, "kv_precision": pair})
    result = export_to_backend(_model(), cfg)
    manifest = json.load(open(result.manifest_path))
    assert manifest["kv_precision"] == list(pair)
    assert manifest["cache_impl"] == ("io_rows" if "fp8" in pair else "io_inplace")
    inputs = {i.name: i.type.tensor_type for i in onnx.load(result.onnx_path, load_external_data=False).graph.input}
    for which, name in zip(("key", "value"), pair):
        assert inputs[f"past_{which}_0"].elem_type == stored_types[name]
        assert inputs[f"past_{which}_0"].shape.dim[-1].dim_value == KV_PRECISIONS[name].row_width(head_dim)
        assert manifest["caches"][0][which]["dtype"] == \
            str(KV_PRECISIONS[name].storage_dtype(torch.float32)).removeprefix("torch.")

    try:
        session = ort.InferenceSession(result.onnx_path)
    except Exception as exc:  # an onnxruntime without the opset-24 TensorScatter
        pytest.skip(f"onnxruntime cannot load the exported model: {exc}")
    tokens = [3, 17, 42, 5, 99, 7]
    got, caches = _run_onnx(session, manifest, tokens)
    assert caches["past_key_0"].dtype == KV_PRECISIONS[pair[0]].storage_dtype(torch.float32)
    assert caches["past_value_0"].dtype == KV_PRECISIONS[pair[1]].storage_dtype(torch.float32)
    eager = _greedy_logits(_wrapper(model, manifest["cache_impl"], pair), tokens)
    _assert_backend_matches(got, eager, 1e-3, max(ONNX_ERROR[name] for name in pair))


@pytest.fixture
def release_cuda():
    """Give the GPU memory a test held (engines, CUDA graphs, cached blocks) back afterwards, so the cases
    of a GPU test do not add up."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@caps.requires_tensorrt
@pytest.mark.parametrize("name", ["full", "int8", "int4", "fp8", pytest.param("nvfp4", marks=caps.requires_float4),
                                  ("fp8", "int4"), pytest.param(("full", "nvfp4"), marks=caps.requires_float4)])
def test_tensorrt_engine_holds_every_precision_in_a_rows_cache(tmp_path, release_cuda, name):
    """TensorRT with cache_impl="io_rows": the engine reads the stored cache (fp8 through a dequantize
    layer, nvfp4 through two, uint8 through casts) and returns the new rows; the session writes them,
    inside the decode CUDA graph. A prompt (prefill profile) plus decode steps match the eager model. The
    steps are fed the eager model's greedy tokens, so the two cannot drift onto different tokens, and the
    eager model leaves the GPU before the engine is built: one copy of the weights on the GPU at a time."""
    from transformersurgeon.export import build_wrapper, export_to_backend
    from transformersurgeon.export.backends import TensorRTExportConfig
    from transformersurgeon.export.backends.tensorrt.session import TensorRTLLMSession

    options = {"max_cache_len": CACHE, "kv_precision": name, "cache_impl": "io_rows", "rmsnorm_impl": "rmsnorm_upcast"}
    prompt, steps = torch.tensor([3, 17, 42, 5, 99]), 4

    model = _model(torch.float16)
    decoder = convert_for_export(model, options=options)["text"]
    wrapper = build_wrapper({"embedding": model.get_input_embeddings(), "decoder": decoder,
                             "final_layer": model.lm_head}).cuda().eval()
    keys, values = move_to_device(build_zero_caches(wrapper.decoder), torch.device("cuda"))
    ids, pos, expected, tokens = prompt.cuda(), torch.tensor([0]).cuda(), [], []
    with torch.no_grad():
        for _ in range(steps + 1):
            logits, new_keys, new_values = wrapper(ids, pos, keys, values)
            for cache, rows in zip(keys + values, new_keys + new_values):
                write_cache_rows(cache, rows, pos)
            expected.append(logits.float().cpu())
            pos, ids = pos + ids.numel(), logits.float().argmax(-1).view(1)
            tokens.append(int(ids))
    del model, decoder, wrapper, keys, values, logits, new_keys, new_values
    gc.collect()
    torch.cuda.empty_cache()

    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), max_input_len=8, convert_options=options)
    result = export_to_backend(_model(torch.float16), cfg)
    session = TensorRTLLMSession(result.manifest_path)
    assert session.cache_rows
    key, value = kv_precision_pair(name)   # a (key, value) pair stores the two caches differently
    assert session.caches["past_key_0"].dtype == KV_PRECISIONS[key].storage_dtype(torch.float16)
    assert session.caches["past_value_0"].dtype == KV_PRECISIONS[value].storage_dtype(torch.float16)

    session.prefill(prompt)
    got = [session.logits.float().cpu()]
    for token in tokens[:steps]:
        session.input_ids.fill_(token)   # the next step consumes the eager model's token
        session.step()
        got.append(session.logits.float().cpu())
    _assert_backend_matches(torch.stack(got), torch.stack(expected), 3e-2, max(ENGINE_ERROR[key], ENGINE_ERROR[value]))


# --------------------------------------------------------------------------- #
# QNN: an 8-bit fixed-point cache chosen by the QNN quantiser (kv_cache_bits)
# --------------------------------------------------------------------------- #
def _qnn_components(cache_impl="mutable"):
    model = _model()
    decoder = convert_for_export(model, options={"cache_impl": cache_impl, "max_cache_len": CACHE})["text"]
    return {"embedding": model.get_input_embeddings(), "decoder": decoder, "final_layer": model.lm_head,
            "config": model.config}


@pytest.mark.parametrize("cache_impl, per_layer", [("mutable", 2), ("io_scatter", 2), ("io_concat", 0)])
def test_qnn_kv_cache_predicate_selects_the_cache_writes(cache_impl, per_layer):
    """One key and one value write per layer for the caches written by an index operation. io_concat
    writes with a select, which the predicate does not match: the exporter rejects kv_cache_bits=8 with it
    (test_qnn_kv_cache_bits_8_rejects_a_cache_it_cannot_quantise)."""
    pytest.importorskip("torchao")  # imported by the export backends
    from transformersurgeon.export.backends.qnn.passes import (
        _decompose_wrap_with_set_grad_enabled, _kv_cache_write_kind, _kv_cache_write_predicate,
    )
    from transformersurgeon.export.wrapper import resolve_components_and_wrapper

    wrapper, _, example_inputs = resolve_components_and_wrapper(_qnn_components(cache_impl))
    with torch.inference_mode():  # as the QNN exporter traces it
        module = torch.export.export(wrapper, example_inputs).module()
    try:
        _decompose_wrap_with_set_grad_enabled(module)  # as the QNN exporter does (needs ExecuTorch)
    except ImportError:
        pass  # an inference-mode trace has no grad-mode wrapper to inline
    predicate = _kv_cache_write_predicate()
    writes = [node for node in module.graph.nodes if predicate(node)]
    assert len(writes) == per_layer * len(wrapper.decoder.blocks)
    # Each write says which cache it updates: kv_cache_bits=(16, 8) quantises the value writes only.
    kinds = [_kv_cache_write_kind(node) for node in writes]
    assert kinds.count("key") == kinds.count("value") == len(writes) // 2
    values = [node for node in module.graph.nodes if _kv_cache_write_predicate(("value",))(node)]
    assert values == [node for node, kind in zip(writes, kinds) if kind == "value"]


def test_qnn_kv_cache_bits_needs_a_fully_quantised_graph():
    pytest.importorskip("torchao")
    from transformersurgeon.export.backends import QNNExportConfig
    from transformersurgeon.export.backends.qnn.exporter import QNNExporter
    from transformersurgeon.export.operations import check_config_choices

    assert QNNExportConfig(output_path="x").kv_cache_bits == 16
    assert QNNExportConfig(output_path="x", kv_cache_bits=[16, 8]).kv_cache_bits == (16, 8)  # as read from JSON
    for bits in (8, (16, 8)):
        check_config_choices("qnn", QNNExportConfig(output_path="x", kv_cache_bits=bits))
    assert QNNExportConfig(output_path="x", kv_cache_bits=(8, 8)).kv_cache_bits == 8   # one number for both caches
    for bits in ((8, 16), (16, 16)):
        check_config_choices("qnn", QNNExportConfig(output_path="x", kv_cache_bits=bits))
    for bits in (4, (4, 8)):
        with pytest.raises(ValueError, match="not supported by the qnn backend"):
            check_config_choices("qnn", QNNExportConfig(output_path="x", kv_cache_bits=bits))
    for bits in (8, (16, 8)):
        for bad in ({"quant_scope": "linear"}, {"quant_scope": "all", "float_attention_core": True}):
            config = QNNExportConfig(output_path="x", kv_cache_bits=bits, **bad)
            with pytest.raises(ValueError, match=re.escape(f"kv_cache_bits={bits!r} requires quant_scope='all'")):
                QNNExporter().lower(_qnn_components(), config)


@caps.requires_qnn
def test_qnn_kv_cache_bits_8_quantises_the_cache_writes_to_8_bits(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import QNNExportConfig

    cfg = QNNExportConfig(output_path=str(tmp_path / "model.pte"), max_seq_len=CACHE, quant_scope="all",
                          kv_cache_bits=8, keep_reference_module=True, run_weight_mismatch_check=False,
                          convert_options={"max_cache_len": CACHE})
    result = export_to_backend(_model(), cfg)
    assert os.path.getsize(cfg.output_path) > 0
    # In the fake-quant model the .pte was lowered from, what each cache write stores is
    # dequantised from an 8-bit range (16-bit everywhere else).
    graph = result.reference_module.graph
    writes = [n for n in graph.nodes if n.op == "call_function" and "index_copy" in str(n.target)]
    ranges = [w.args[3].args[4] for w in writes
              if isinstance(w.args[3], torch.fx.Node) and "dequantize" in str(w.args[3].target)]
    assert writes and ranges and max(ranges) <= 255


@caps.requires_qnn
@pytest.mark.parametrize("bits, cache_impl", [((16, 8), "mutable"), ((16, 8), "io_scatter"), ((8, 16), "mutable")])
def test_qnn_kv_cache_bits_pair_sets_each_cache(tmp_path, bits, cache_impl):
    """(key bits, value bits): each cache gets its own fixed-point width, e.g. (16, 8) an 8-bit value cache and a
    16-bit key cache (an 8-bit key cache loses accuracy: a few key channels with large biases set its per-tensor
    step, AGENTS.md, KV cache)."""
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import QNNExportConfig
    from transformersurgeon.export.backends.qnn.passes import _KV_CACHE_WRITES, _kv_cache_write_kind

    model = _model()
    cfg = QNNExportConfig(output_path=str(tmp_path / "model.pte"), max_seq_len=CACHE, quant_scope="all",
                          kv_cache_bits=bits, keep_reference_module=True, run_weight_mismatch_check=False,
                          convert_options={"max_cache_len": CACHE, "cache_impl": cache_impl})
    graph = export_to_backend(model, cfg).reference_module.graph
    # In the fake-quant model the .pte was lowered from: the integer range each cache write stores.
    levels = {"key": [], "value": []}
    for write in [n for n in graph.nodes if n.op == "call_function" and n.target in _KV_CACHE_WRITES]:
        stored = write.args[3] if "index_copy" in str(write.target) else write.args[2]
        assert "dequantize" in str(stored.target)
        levels[_kv_cache_write_kind(write)].append(stored.args[4])
    assert len(levels["key"]) == len(levels["value"]) == model.config.num_hidden_layers
    for which, cache_bits in zip(("key", "value"), bits):
        assert set(levels[which]) == {2 ** cache_bits - 1}


@caps.requires_qnn
def test_qnn_kv_cache_bits_8_rejects_a_cache_it_cannot_quantise(tmp_path):
    """io_concat is a QNN cache, but nothing in it matches the cache-write predicate: without the check the
    export would succeed with a 16-bit cache."""
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import QNNExportConfig

    cfg = QNNExportConfig(output_path=str(tmp_path / "model.pte"), max_seq_len=CACHE, quant_scope="all",
                          kv_cache_bits=8, run_weight_mismatch_check=False,
                          convert_options={"max_cache_len": CACHE, "cache_impl": "io_concat"})
    with pytest.raises(ValueError, match="found no KV-cache write to quantise"):
        export_to_backend(_model(), cfg)
    assert not os.path.exists(cfg.output_path)
