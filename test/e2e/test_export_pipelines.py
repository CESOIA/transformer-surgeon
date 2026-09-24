"""End-to-end export-pipeline tests: load -> compress -> export, one per backend.

These use a real (small) cached checkpoint because the export lowering paths depend
on genuine weights/shapes. Every backend is guarded by a capability marker so the
module degrades gracefully:

    HF roundtrip   -> always runnable (needs Hub access)
    convert graph  -> always runnable
    XNNPACK (.pte) -> requires the `executorch` extra
    TensorRT       -> requires onnx, the `tensorrt` package + CUDA (ONNX -> engine)
    QNN (.pte)     -> requires the Qualcomm QNN SDK (skipped in most environments)

All exports are marked ``slow`` (they compile/lower a real graph).
"""
from __future__ import annotations

import os

import pytest
import torch

from _helpers import capabilities as caps

pytestmark = [pytest.mark.e2e, pytest.mark.download, pytest.mark.slow, caps.requires_hub]

MODEL_NAME = "Qwen/Qwen2-0.5B"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qwen_classes():
    from transformersurgeon import (
        Qwen2ForCausalLMCompress,
        Qwen2CompressionSchemesManager,
        convert_for_export,
    )

    return Qwen2ForCausalLMCompress, Qwen2CompressionSchemesManager, convert_for_export


def _load_float_model(ModelCls):
    """Load the checkpoint on CPU in float16. Kept on CPU: the backend exporters
    build fake-tensor example inputs on CPU and place the graph on the target
    device themselves (moving the whole model to CUDA first breaks torch.export —
    see docs/investigations/FRAMEWORK_PROBLEMS.md #4)."""
    try:
        return ModelCls.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).eval()
    except Exception as exc:  # network / auth failure -> skip, don't fail
        pytest.skip(f"could not load {MODEL_NAME}: {exc}")


def _quantize_two_mlp_layers(model, ManagerCls):
    mgr = ManagerCls(model)
    mgr.set("quantization", "precision", 8,
            criteria=[[0, "mlp.gate_proj"], [0, "mlp.up_proj"]])
    mgr.apply(hard=True)
    model.eval()
    return model


def _components(model, convert_for_export):
    converted = convert_for_export(model, options={"use_sdpa": False})
    return {
        "embedding": model.get_input_embeddings(),
        "decoder": converted["text"],
        "final_layer": model.lm_head,
        "config": model.config,
    }


# --------------------------------------------------------------------------- #
# Backend-agnostic paths
# --------------------------------------------------------------------------- #
def test_hf_export_roundtrip(qwen_classes, out_dir):
    """Compress with LRD, export to a local HF dir via the manager, reload, forward."""
    ModelCls, ManagerCls, _ = qwen_classes
    from transformersurgeon.hf import export_to_hf

    model = _load_float_model(ModelCls)
    mgr = ManagerCls(model)
    mgr.set("lrd", "rank", 64, criteria=[[0, "mlp.down_proj"]])

    export_to_hf(model, repo_id="test/roundtrip", out_dir=out_dir, manager=mgr, exist_ok=True)
    saved = os.path.join(out_dir, "roundtrip")
    assert os.path.isdir(saved)

    reloaded = ModelCls.from_pretrained(saved, torch_dtype="auto").eval()
    with torch.no_grad():
        out = reloaded(input_ids=torch.randint(0, 100, (1, 4)))
    assert out.logits is not None


def test_convert_for_export_graph(qwen_classes):
    ModelCls, _, convert_for_export = qwen_classes
    model = _load_float_model(ModelCls)
    converted = convert_for_export(model, options={"use_sdpa": False})
    assert "text" in converted


# --------------------------------------------------------------------------- #
# Deployment backends
# --------------------------------------------------------------------------- #
@caps.requires_executorch
def test_export_xnnpack(qwen_classes, out_dir):
    ModelCls, ManagerCls, convert_for_export = qwen_classes
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.executorch_exporters.xnnpack import XNNPACKExportConfig

    model = _quantize_two_mlp_layers(_load_float_model(ModelCls), ManagerCls)
    comps = _components(model, convert_for_export)

    pte = os.path.join(out_dir, "qwen2_xnnpack.pte")
    cfg = XNNPACKExportConfig(output_path=pte, backend="xnnpack", max_seq_len=128,
                              convert_options={"use_sdpa": False},
                              run_weight_mismatch_check=False, verbose=False)
    export_to_backend(comps, config=cfg)
    assert os.path.isfile(pte) and os.path.getsize(pte) > 0


@caps.requires_executorch
@caps.requires_custom_sdpa
def test_export_xnnpack_custom_sdpa(qwen_classes, out_dir):
    """Same as test_export_xnnpack, but with attn_impl='custom_sdpa' -- ExecuTorch's
    fused CPU custom ops (torch.ops.llama.custom_sdpa/update_cache) instead of the
    default manual attention kernel."""
    ModelCls, ManagerCls, convert_for_export = qwen_classes
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.executorch_exporters.xnnpack import XNNPACKExportConfig

    model = _quantize_two_mlp_layers(_load_float_model(ModelCls), ManagerCls)
    converted = convert_for_export(
        model, options={"use_sdpa": False, "attn_impl": "custom_sdpa", "cache_impl": "mutable"}
    )
    comps = {
        "embedding": model.get_input_embeddings(),
        "decoder": converted["text"],
        "final_layer": model.lm_head,
        "config": model.config,
    }

    pte = os.path.join(out_dir, "qwen2_xnnpack_custom_sdpa.pte")
    cfg = XNNPACKExportConfig(output_path=pte, backend="xnnpack", max_seq_len=128,
                              convert_options={"use_sdpa": False, "attn_impl": "custom_sdpa", "cache_impl": "mutable"},
                              run_weight_mismatch_check=False, verbose=False)
    export_to_backend(comps, config=cfg)
    assert os.path.isfile(pte) and os.path.getsize(pte) > 0


@caps.requires_tensorrt
@pytest.mark.parametrize("int4_backend", [None, "dequantize", "edgellm_plugin"])
def test_export_tensorrt(qwen_classes, out_dir, int4_backend):
    """ONNX -> plain TensorRT (the Jetson path): in-place KV cache, prefill +
    CUDA-graph decode through TensorRTLLMSession, greedy tokens vs eager."""
    if int4_backend == "edgellm_plugin" and not caps.HAS_EDGELLM_PLUGIN:
        pytest.skip("requires tensorrt_edgellm and EDGELLM_PLUGIN_PATH")
    ModelCls, ManagerCls, convert_for_export = qwen_classes
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.common import build_wrapper, build_zero_caches
    from transformersurgeon.export.tensorrt import TensorRTExportConfig
    from transformersurgeon.export.tensorrt.session import TensorRTLLMSession

    model = _load_float_model(ModelCls)
    if int4_backend is not None:
        mgr = ManagerCls(model)
        crit = ["self_attn", "mlp"]
        mgr.set("quantization", "precision", 4, criteria=crit)
        mgr.set("quantization", "granularity", "per_channel", criteria=crit)
        mgr.apply(hard=True, criteria=crit)
    options = {"use_sdpa": False, "cache_impl": "io_inplace", "max_cache_len": 128,
               "rmsnorm_prescale": False, "rmsnorm_upcast": True}
    cfg = TensorRTExportConfig(output_path=os.path.join(out_dir, "model.onnx"), backend="tensorrt",
                                   max_input_len=32, convert_options=options,
                                   int4_backend=int4_backend or "dequantize")
    result = export_to_backend(model, config=cfg)
    assert os.path.isfile(result.engine_path)

    session = TensorRTLLMSession(result.manifest_path)
    prompt = torch.tensor([9707, 11, 1246, 525, 498, 30])  # "Hello, how are you?"
    tokens = session.generate(prompt, 8, stop_at_eos=False)
    assert len(tokens) == 8

    # Eager reference for the first token (same conversion options).
    ref_model = _load_float_model(ModelCls)
    if int4_backend is not None:
        mgr = ManagerCls(ref_model)
        mgr.set("quantization", "precision", 4, criteria=crit)
        mgr.set("quantization", "granularity", "per_channel", criteria=crit)
        mgr.apply(hard=True, criteria=crit)
    wrapper = build_wrapper({"embedding": ref_model.get_input_embeddings(),
                             "decoder": convert_for_export(ref_model, options=options)["text"],
                             "final_layer": ref_model.lm_head}, model_config=ref_model.config).cuda().eval()
    from transformersurgeon.export.common import extract_layer_quant_info
    extract_layer_quant_info(wrapper)  # same dequantized weights the exporter sees
    keys, values = build_zero_caches(wrapper.decoder)
    with torch.no_grad():
        ref = wrapper(prompt.cuda(), torch.tensor([0]).cuda(), [k.cuda() for k in keys], [v.cuda() for v in values])[0]
    assert tokens[0] == int(ref.argmax())


@caps.requires_qnn
def test_export_qnn(qwen_classes, out_dir):
    ModelCls, ManagerCls, convert_for_export = qwen_classes
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.executorch_exporters.qnn import QNNExportConfig

    model = _quantize_two_mlp_layers(_load_float_model(ModelCls), ManagerCls)
    comps = _components(model, convert_for_export)

    pte = os.path.join(out_dir, "qwen2_qnn.pte")
    cfg = QNNExportConfig(output_path=pte, backend="qnn", max_seq_len=128,
                          convert_options={"use_sdpa": False},
                          run_weight_mismatch_check=False, verbose=False)
    export_to_backend(comps, config=cfg)
    assert os.path.isfile(pte) and os.path.getsize(pte) > 0
