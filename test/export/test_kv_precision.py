"""KV-cache precision through the export pipeline: options, conversion, wrapper, manifest, backends."""
import json
import os

import pytest
import torch

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES
from transformersurgeon.export import (
    LLMWrapper, build_llm_manifest, build_zero_caches, convert_for_export, zero_caches_from_manifest,
)
from transformersurgeon.export.operations import check_components, options_for, resolve_convert_options

CACHE = 16


def _model():
    torch.manual_seed(0)
    return FAMILIES["qwen2"].build().float().eval()


def _wrapper(model, cache_impl="mutable", kv_precision="full"):
    options = {"cache_impl": cache_impl, "max_cache_len": CACHE, "kv_precision": kv_precision}
    decoder = convert_for_export(model, options=options)["text"]
    return LLMWrapper(model.get_input_embeddings(), decoder, model.lm_head).eval()


def _greedy_logits(wrapper, tokens):
    """Logits of every step of a token-by-token run, with the wrapper's own caches."""
    keys, values = build_zero_caches(wrapper.decoder)
    out = []
    with torch.no_grad():
        for t, token in enumerate(tokens):
            args = (torch.tensor([token]), torch.tensor([t]))
            if wrapper.graph_io_cache:
                logits, keys, values = wrapper(*args, keys, values)
            else:
                logits = wrapper(*args)
            out.append(logits)
    return torch.stack(out)


@pytest.mark.parametrize("backend, supported", [
    ("xnnpack", {"full", "int8", "int4"}), ("onnx", {"full", "int8", "int4"}),
    ("qnn", {"full"}), ("tensorrt", {"full"}),
])
def test_backends_list_the_precisions_they_support(backend, supported):
    assert set(options_for(backend, "kv_precision")) == supported
    assert resolve_convert_options(backend)["kv_precision"] == "full"
    for name in supported:
        assert resolve_convert_options(backend, {"kv_precision": name})["kv_precision"] == name
    if "int8" not in supported:
        with pytest.raises(ValueError, match=f"not supported by the {backend} backend"):
            resolve_convert_options(backend, {"kv_precision": "int8"})


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
