"""KV-cache precision through the export pipeline: options, conversion, wrapper, manifest, backends."""
import json
import os

import pytest
import torch

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES
from transformersurgeon.blocks import KV_PRECISIONS
from transformersurgeon.export import (
    LLMWrapper, build_llm_manifest, build_zero_caches, convert_for_export, write_cache_rows,
    zero_caches_from_manifest,
)
from transformersurgeon.export.operations import check_components, options_for, resolve_convert_options
from transformersurgeon.utils.devices import move_to_device

CACHE = 16


def _model(dtype=torch.float32):
    torch.manual_seed(0)
    model = FAMILIES["qwen2"].build().to(dtype).eval()
    model.config.dtype = dtype  # the converted decoder allocates its cache in the config dtype
    return model


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


@pytest.mark.parametrize("cache_impl", ["mutable", "io_inplace"])
@pytest.mark.parametrize("name, tolerance", [("int8", 0.02), ("int4", 0.25)])
def test_integer_cache_wrapper_tracks_the_float_wrapper(cache_impl, name, tolerance):
    model = _model()
    tokens = [3, 17, 42, 5, 99, 7]
    expected = _greedy_logits(_wrapper(model, cache_impl), tokens)
    got = _greedy_logits(_wrapper(model, cache_impl, name), tokens)
    assert (got - expected).norm() / expected.norm() < tolerance


def test_manifest_records_the_precision_and_gives_uint8_caches(tmp_path):
    model = _model()
    wrapper = _wrapper(model, "io_inplace", "int8")
    manifest = build_llm_manifest(wrapper, model.config, backend="onnx", artifact=str(tmp_path / "model.onnx"))
    assert manifest["kv_precision"] == "int8" and manifest["cache_dtype"] == "uint8"
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    assert manifest["caches"][0]["key"]["shape"] == [1, 2, CACHE, head_dim + 2]
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
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    assert manifest["caches"][0]["key"]["shape"] == [1, 2, CACHE, head_dim]
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
    head_dim = model.config.hidden_size // model.config.num_attention_heads   # 16: one block per row
    layer = manifest["caches"][1]
    assert layer["key"] == {"input": "past_key_1", "output": "present_key_1", "shape": [1, 2, CACHE, head_dim // 2],
                            "dtype": "float4_e2m1fn_x2"}
    assert layer["value_scale"] == {"input": "past_value_scale_1", "output": "present_value_scale_1",
                                    "shape": [1, 2, CACHE, head_dim // 16], "dtype": "float8_e4m3fn"}
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
    result = export_to_backend(_model(torch.float16), cfg)
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    inputs, outputs = graph_io_names(2, scales=True)
    assert [i.name for i in graph.input] == inputs and [o.name for o in graph.output] == outputs
    types = {i.name: (i.type.tensor_type.elem_type, [d.dim_value for d in i.type.tensor_type.shape.dim])
             for i in graph.input}
    assert types["past_key_0"] == (onnx.TensorProto.FLOAT4E2M1, [1, 2, CACHE, 16])
    assert types["past_value_scale_1"] == (onnx.TensorProto.FLOAT8E4M3FN, [1, 2, CACHE, 1])
    fp4_quantize = [n for n in graph.node if n.op_type == "QuantizeLinear"
                    and onnx.helper.get_attribute_value(next(a for a in n.attribute if a.name == "output_dtype"))
                    == onnx.TensorProto.FLOAT4E2M1]
    assert len(fp4_quantize) == 4   # keys and values of two layers
    assert all(next(a.i for a in n.attribute if a.name == "block_size") == 16 for n in fp4_quantize)
    assert json.load(open(result.manifest_path))["cache_impl"] == "io_rows"


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
    torch.testing.assert_close(torch.from_numpy(logits).reshape(expected.shape), expected, atol=1e-3, rtol=1e-3)


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
    head_dim = model.config.hidden_size // model.config.num_attention_heads
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
        torch.testing.assert_close(logits[name], eager, atol=1e-3, rtol=1e-3)

    def distance(name):
        return ((logits[name] - logits["full"]).norm() / logits["full"].norm()).item()

    assert 0 < distance("int8") < distance("int4") < 0.25
    assert 0 < distance("fp8") < distance("int4")


@caps.requires_tensorrt
@pytest.mark.parametrize("name", ["full", "int8", "int4", "fp8", pytest.param("nvfp4", marks=caps.requires_float4)])
def test_tensorrt_engine_holds_every_precision_in_a_rows_cache(tmp_path, name):
    """TensorRT with cache_impl="io_rows": the engine reads the stored cache (fp8 through a dequantize
    layer, nvfp4 through two, uint8 through casts) and returns the new rows; the session writes them,
    inside the decode CUDA graph. A prompt (prefill profile) plus greedy steps match the eager model."""
    from transformersurgeon.export import build_wrapper, export_to_backend
    from transformersurgeon.export.backends import TensorRTExportConfig
    from transformersurgeon.export.backends.tensorrt.session import TensorRTLLMSession

    options = {"max_cache_len": CACHE, "kv_precision": name, "cache_impl": "io_rows", "rmsnorm_impl": "rmsnorm_upcast"}
    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), max_input_len=8, convert_options=options)
    result = export_to_backend(_model(torch.float16), cfg)
    session = TensorRTLLMSession(result.manifest_path)
    assert session.cache_rows
    assert session.caches["past_key_0"].dtype == KV_PRECISIONS[name].storage_dtype(torch.float16)

    prompt, steps = torch.tensor([3, 17, 42, 5, 99]), 4
    session.prefill(prompt)
    got = [session.logits.float().clone()]
    for _ in range(steps):
        session.step()
        got.append(session.logits.float().clone())

    model = _model(torch.float16)
    decoder = convert_for_export(model, options=options)["text"]
    wrapper = build_wrapper({"embedding": model.get_input_embeddings(), "decoder": decoder,
                             "final_layer": model.lm_head}).cuda().eval()
    keys, values = move_to_device(build_zero_caches(wrapper.decoder), torch.device("cuda"))
    ids, pos, expected = prompt.cuda(), torch.tensor([0]).cuda(), []
    with torch.no_grad():
        for _ in range(steps + 1):
            logits, new_keys, new_values = wrapper(ids, pos, keys, values)
            for cache, rows in zip(keys + values, new_keys + new_values):
                write_cache_rows(cache, rows, pos)
            expected.append(logits.float())
            pos, ids = pos + ids.numel(), logits.float().argmax(-1).view(1)
    torch.testing.assert_close(torch.stack(got), torch.stack(expected), atol=3e-2, rtol=3e-2)


# --------------------------------------------------------------------------- #
# QNN: an 8-bit fixed-point cache chosen by the QNN quantiser (kv_cache_bits)
# --------------------------------------------------------------------------- #
def _qnn_components(cache_impl="mutable"):
    model = _model()
    decoder = convert_for_export(model, options={"cache_impl": cache_impl, "max_cache_len": CACHE})["text"]
    return {"embedding": model.get_input_embeddings(), "decoder": decoder, "final_layer": model.lm_head,
            "config": model.config}


@pytest.mark.parametrize("cache_impl", ["mutable", "io_scatter"])
def test_qnn_kv_cache_predicate_selects_the_cache_writes(cache_impl):
    pytest.importorskip("torchao")  # imported by the export backends
    from transformersurgeon.export.backends.qnn.passes import (
        _decompose_wrap_with_set_grad_enabled, _kv_cache_write_predicate,
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
    assert len(writes) == 2 * len(wrapper.decoder.blocks)  # one key and one value write per layer


def test_qnn_kv_cache_bits_needs_a_fully_quantised_graph():
    pytest.importorskip("torchao")
    from transformersurgeon.export.backends import QNNExportConfig
    from transformersurgeon.export.backends.qnn.exporter import QNNExporter
    from transformersurgeon.export.operations import check_config_choices

    assert QNNExportConfig(output_path="x").kv_cache_bits == 16
    check_config_choices("qnn", QNNExportConfig(output_path="x", kv_cache_bits=8))
    with pytest.raises(ValueError, match="not supported by the qnn backend"):
        check_config_choices("qnn", QNNExportConfig(output_path="x", kv_cache_bits=4))
    for bad in ({"quant_scope": "linear"}, {"quant_scope": "all", "float_attention_core": True}):
        config = QNNExportConfig(output_path="x", kv_cache_bits=8, **bad)
        with pytest.raises(ValueError, match="kv_cache_bits=8 requires quant_scope='all'"):
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
