"""export/exporter.py and export/backends/: registry, real exports (capability-gated), graph passes."""
import json
import os

import numpy as np
import pytest
import torch

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES
from transformersurgeon.export import (
    EXPORTERS, BackendExportConfig, build_wrapper, build_zero_caches, convert_for_export, export_to_backend,
    extract_layer_quant_info, get_exporter,
)
from transformersurgeon.export.exporter import resolve_components

CACHE = 16
IO_INPLACE = {"cache_impl": "io_inplace", "max_cache_len": CACHE}


def _model(dtype=torch.float32, quantize_mlp=False):
    torch.manual_seed(0)
    spec = FAMILIES["qwen2"]
    model = spec.build().to(dtype).eval()
    model.config.dtype = dtype  # the converted decoder allocates its cache in the config dtype
    if quantize_mlp:
        mgr = spec.manager(model)
        mgr.set("quantization", "precision", 8, criteria="mlp")
        mgr.set("quantization", "granularity", "per_channel", criteria="mlp")
        mgr.apply(hard=True, show_progress=False)
    return model


def _components(model, options):
    return {"embedding": model.get_input_embeddings(), "decoder": convert_for_export(model, options=options)["text"],
            "final_layer": model.lm_head, "config": model.config}


# --------------------------------------------------------------------------- #
# Registry and component resolution
# --------------------------------------------------------------------------- #
def test_every_backend_is_registered():
    get_exporter("onnx")  # importing the backends registers them
    assert set(EXPORTERS) == {"xnnpack", "qnn", "onnx", "tensorrt"}
    with pytest.raises(ValueError, match="Unknown export backend"):
        get_exporter("nope")


def test_exporter_rejects_a_config_of_another_backend(tmp_path):
    from transformersurgeon.export.backends import XNNPACKExportConfig

    with pytest.raises(TypeError, match="ONNXExportConfig"):
        get_exporter("onnx").export(_model(), XNNPACKExportConfig(output_path=str(tmp_path / "m.pte")))
    with pytest.raises(ValueError, match="config"):
        export_to_backend(_model(), None)
    assert issubclass(XNNPACKExportConfig, BackendExportConfig)


def test_resolve_components_accepts_model_dict_and_tuple():
    model = _model()
    emb, decoder, head, config = resolve_components(model, convert_options=IO_INPLACE)
    assert emb is model.get_input_embeddings() and head is model.lm_head and config is model.config
    assert decoder.cache_impl == "io_inplace" and len(decoder.blocks) == 2
    assert resolve_components((emb, decoder, head)) == (emb, decoder, head, None)
    assert resolve_components({"embedding": emb, "decoder": decoder, "final_layer": head})[3] is None
    with pytest.raises(TypeError):
        resolve_components(object())


# --------------------------------------------------------------------------- #
# Real exports
# --------------------------------------------------------------------------- #
@caps.requires_onnx
def test_onnx_export_uses_the_backend_default_cache_and_writes_the_manifest(tmp_path):
    import onnx

    from transformersurgeon.export.backends import ONNXExportConfig

    cfg = ONNXExportConfig(output_path=str(tmp_path / "model.onnx"), max_input_len=8,
                           convert_options={"max_cache_len": CACHE})
    result = export_to_backend(_model(), cfg)
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    assert [n.op_type for n in graph.node].count("TensorScatter") == 4  # 2 layers x (key, value)
    assert [i.name for i in graph.input][:4] == ["input_ids", "pos_id", "past_key_0", "past_key_1"]
    manifest = json.load(open(result.manifest_path))
    assert manifest["cache_impl"] == "io_inplace" and manifest["max_input_len"] == 8
    assert manifest["caches"][0]["key"]["shape"] == [1, 2, CACHE, 16]


@caps.requires_onnx
def test_onnx_export_rejects_an_unsupported_cache(tmp_path):
    from transformersurgeon.export.backends import ONNXExportConfig

    cfg = ONNXExportConfig(output_path=str(tmp_path / "m.onnx"), convert_options={"cache_impl": "mutable"})
    with pytest.raises(ValueError, match="mutable"):
        export_to_backend(_model(), cfg)


@caps.requires_tensorrt
def test_tensorrt_engine_matches_the_eager_wrapper(tmp_path):
    from transformersurgeon.export.backends import TensorRTExportConfig
    from transformersurgeon.export.backends.tensorrt.session import TensorRTLLMSession

    options = {**IO_INPLACE, "rmsnorm_impl": "rmsnorm_upcast"}
    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), max_input_len=8, convert_options=options)
    result = export_to_backend(_model(torch.float16), cfg)
    assert os.path.isfile(result.engine_path)

    prompt = torch.tensor([3, 17, 42, 5, 99])
    session = TensorRTLLMSession(result.manifest_path)
    session.prefill(prompt)
    wrapper = build_wrapper(_components(_model(torch.float16), options)).cuda().eval()
    keys, values = build_zero_caches(wrapper.decoder)
    with torch.no_grad():
        ref = wrapper(prompt.cuda(), torch.tensor([0]).cuda(), [k.cuda() for k in keys], [v.cuda() for v in values])[0]
    torch.testing.assert_close(session.logits.float(), ref.float(), atol=3e-2, rtol=3e-2)


@caps.requires_executorch
def test_xnnpack_export_writes_a_program(tmp_path):
    from transformersurgeon.export.backends import XNNPACKExportConfig

    pte = str(tmp_path / "model.pte")
    cfg = XNNPACKExportConfig(output_path=pte, max_seq_len=CACHE, run_weight_mismatch_check=False, verbose=False)
    result = export_to_backend(_components(_model(quantize_mlp=True), {"max_cache_len": CACHE}), cfg)
    assert os.path.getsize(pte) > 0 and result.pte_path == pte and os.path.isfile(result.manifest_path)


@caps.requires_qnn
def test_qnn_export_writes_a_program(tmp_path):
    from transformersurgeon.export.backends import QNNExportConfig

    pte = str(tmp_path / "model.pte")
    # Whole-graph quantization (vendor recipe): the float/quantized seams of the
    # default scope do not all lower on the HTP of this toolchain version.
    cfg = QNNExportConfig(output_path=pte, soc_model="SM8850", max_seq_len=CACHE, quant_scope="all",
                          calibration_token_ids=[[(i * 37 + j * 11) % 256 for j in range(8)] for i in range(2)],
                          run_weight_mismatch_check=False, verbose=False)
    result = export_to_backend(_components(_model(quantize_mlp=True), {"max_cache_len": CACHE}), cfg)
    assert os.path.getsize(pte) > 0 and os.path.isfile(result.manifest_path)


@caps.requires_qnn
def test_qnn_quantize_float_model_requires_quant_scope_all(tmp_path):
    from transformersurgeon.export.backends import QNNExportConfig

    cfg = QNNExportConfig(output_path=str(tmp_path / "m.pte"), soc_model="SM8850", max_seq_len=CACHE,
                          quant_scope="linear", quantize_float_model=True, run_weight_mismatch_check=False,
                          verbose=False)
    with pytest.raises(ValueError, match="quantize_float_model"):
        export_to_backend(_components(_model(), {"max_cache_len": CACHE}), cfg)


@caps.requires_qnn
def test_qnn_quantize_float_model_lowers_an_uncompressed_model_as_fixed_point(tmp_path):
    from transformersurgeon.export.backends import QNNExportConfig

    # Activation-only reference: no compression-quantized linear, yet the graph is 16a16w everywhere.
    options = {"max_cache_len": CACHE, "use_sdpa": False, "masked_softmax": True}
    tokens = [5, 17, 99, 3, 42, 200, 7, 64]

    def logits(module):
        out = []
        with torch.no_grad():
            for pos, tok in enumerate(tokens):
                o = module(torch.tensor([tok]), torch.tensor([pos]))
                out.append((o[0] if isinstance(o, (tuple, list)) else o).float().reshape(-1))
        return torch.stack(out)

    ref = logits(build_wrapper(_components(_model(), options)).eval())
    cfg = QNNExportConfig(output_path=str(tmp_path / "m.pte"), soc_model="SM8850", max_seq_len=CACHE,
                          convert_options=options, quant_scope="all", quantize_float_model=True,
                          keep_reference_module=True,
                          calibration_token_ids=[[(i * 37 + j * 11) % 256 for j in range(CACHE)] for i in range(4)],
                          run_weight_mismatch_check=False, verbose=False)
    gm = export_to_backend(_components(_model(), options), cfg).reference_module
    targets = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
    assert any("quantize_per_tensor" in t for t in targets), "no activation fake-quant in the reference"
    assert any("quantize_per_channel" in t for t in targets), "no 16-bit weight fake-quant in the reference"
    fq = logits(gm)
    assert bool((fq.argmax(-1) == ref.argmax(-1)).all()) and (fq - ref).abs().max().item() < 0.1


# --------------------------------------------------------------------------- #
# Backend graph passes
# --------------------------------------------------------------------------- #
def test_qnn_split_float_linear_bias_moves_only_float_biases_and_keeps_outputs():
    pytest.importorskip("torchao")
    pytest.importorskip("executorch")
    from torch.export import export as torch_export

    from transformersurgeon.export.backends.qnn.passes import (
        _decompose_wrap_with_set_grad_enabled, _split_float_linear_bias,
    )

    wrapper = build_wrapper(_components(_model(quantize_mlp=True), {"max_cache_len": CACHE})).eval()
    layer_info = extract_layer_quant_info(wrapper)
    inputs = (torch.tensor([3]), torch.tensor([0]))
    with torch.inference_mode():
        gm = torch_export(wrapper, inputs).module()
    _decompose_wrap_with_set_grad_enabled(gm)
    with torch.no_grad():
        before = gm(*inputs).clone()

    split = _split_float_linear_bias(gm, frozenset(layer_info))
    # Only q/k/v carry a bias, and they are not quantized; the MLP linears are left alone.
    assert sorted(split) == sorted(f"decoder.blocks.{i}.attn.{p}" for i in range(2) for p in ("q_proj", "k_proj", "v_proj"))
    for name, buf in gm.named_buffers():  # same empty KV state for the second run
        if name.endswith("_cache"):
            buf.zero_()
    with torch.no_grad():
        torch.testing.assert_close(gm(*inputs), before, rtol=1e-5, atol=1e-5)


def _onnx_graph(nodes, inits, x_shape, y_shape):
    from onnx import TensorProto, helper

    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, x_shape)],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT16, y_shape)], inits)
    return helper.make_model(graph)


@caps.requires_onnx
def test_fp32_accumulate_lrd_only_touches_low_rank_gemms():
    from onnx import TensorProto, helper, numpy_helper

    from transformersurgeon.export.backends.tensorrt.precision_passes import fp32_accumulate_lrd

    f16 = lambda name, shape: numpy_helper.from_array(np.zeros(shape, np.float16), name)
    model = _onnx_graph(
        [helper.make_node("Gemm", ["x", "a.linear_V.weight"], ["h"], transB=1, name="a_v"),
         helper.make_node("Gemm", ["h", "a.weight"], ["z"], transB=1, name="a_u"),
         helper.make_node("Gemm", ["z", "b.weight"], ["y"], transB=1, name="b")],
        [f16("a.linear_V.weight", (4, 8)), f16("a.weight", (8, 4)), f16("b.weight", (8, 8))], [1, 8], [1, 8])
    manifest = {}
    fp32_accumulate_lrd(model, manifest)
    assert manifest["graph_passes"] == [{"name": "fp32_accumulate_lrd", "gemms": 2}]
    gemms = {n.name: n for n in model.graph.node if n.op_type == "Gemm"}
    assert all("__acc32_" in i for n in ("a_v", "a_u") for i in gemms[n].input)
    assert list(gemms["b"].input) == ["z", "b.weight"]
    assert all(i.data_type == TensorProto.FLOAT16 for i in model.graph.initializer)


@caps.requires_onnx
def test_fp32_attention_scores_wraps_the_score_path_in_casts():
    import onnx
    from onnx import TensorProto, helper

    from transformersurgeon.export.backends.tensorrt.attention_passes import fp32_attention_scores

    model = _onnx_graph(
        [helper.make_node("MatMul", ["x", "x"], ["s"], name="mm"),
         helper.make_node("Add", ["s", "x"], ["m"], name="add"),
         helper.make_node("Softmax", ["m"], ["y"], name="sm")], [], [4, 4], [4, 4])
    manifest = {}
    fp32_attention_scores(model, manifest)
    assert manifest["graph_passes"] == [{"name": "fp32_attention_scores", "patterns": 1}]
    casts = [helper.get_attribute_value(n.attribute[0]) for n in model.graph.node if n.op_type == "Cast"]
    assert casts.count(TensorProto.FLOAT) == 3 and casts.count(TensorProto.FLOAT16) == 1
    assert model.graph.node[-1].output[0] == "y"  # consumers still read the fp16 output name
    onnx.checker.check_model(model)
    with pytest.raises(RuntimeError, match="no MatMul"):
        fp32_attention_scores(_onnx_graph([helper.make_node("Relu", ["x"], ["y"])], [], [4], [4]))
