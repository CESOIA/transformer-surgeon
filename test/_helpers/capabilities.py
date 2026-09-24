"""Runtime capability detection for optional test dependencies.

Each ``HAS_*`` flag is evaluated once at import time. ``pytest`` modules turn these
into skip guards, e.g.::

    from test._helpers.capabilities import requires_tensorrt

    @requires_tensorrt
    def test_tensorrt_export(...):
        ...

This keeps a single source of truth for "can this backend run here?" instead of
copy-pasting ``try/import`` blocks into every export test.
"""
from __future__ import annotations

import importlib.util
import os

import pytest


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _qnn_available() -> bool:
    """The QNN ExecuTorch backend needs the Qualcomm QNN SDK.

    There is no clean import probe (the partitioner imports lazily and raises at
    lowering time), so we gate on the SDK env vars the ExecuTorch QNN backend
    documents. This is intentionally conservative: better to skip than to fail a
    machine that simply does not have the Qualcomm toolchain.
    """
    if not _module_available("executorch"):
        return False
    if os.environ.get("QNN_SDK_ROOT") or os.environ.get("QUALCOMM_SDK_ROOT"):
        return True
    return False


HAS_CUDA = _cuda_available()
HAS_EXECUTORCH = _module_available("executorch")
HAS_QNN = _qnn_available()
# ONNX export path (export/onnx): onnx + onnxscript (torch.onnx's dynamo exporter).
HAS_ONNX = _module_available("onnx") and _module_available("onnxscript")
# TensorRT (ONNX -> engine, export/tensorrt/): the `tensorrt` Python package + CUDA.
HAS_TENSORRT = _module_available("tensorrt") and HAS_CUDA
# TensorRT Edge-LLM's plugin library + Python package (opt-in INT4 plugin path).
HAS_EDGELLM_PLUGIN = (
    HAS_TENSORRT
    and _module_available("tensorrt_edgellm")
    and os.path.isfile(os.environ.get("EDGELLM_PLUGIN_PATH", ""))
)
# Narrower than HAS_EXECUTORCH: a minimal ExecuTorch install may lack the LLM
# custom-ops extension (torch.ops.llama.custom_sdpa / update_cache).
HAS_EXECUTORCH_CUSTOM_SDPA = _module_available("executorch.extension.llm.custom_ops.custom_ops")

# Network access to the Hugging Face Hub. Controlled by an env var so CI can force
# offline runs; defaults to "assume reachable" (tests still skip on actual failure).
HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get(
    "TRANSFORMERS_OFFLINE", "0"
) == "1"


requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="requires a CUDA device")
requires_executorch = pytest.mark.skipif(
    not HAS_EXECUTORCH, reason="requires the `executorch` extra"
)
requires_onnx = pytest.mark.skipif(not HAS_ONNX, reason="requires onnx + onnxscript")
requires_tensorrt = pytest.mark.skipif(
    not (HAS_ONNX and HAS_TENSORRT), reason="requires onnx, the `tensorrt` package and a CUDA device"
)
requires_edgellm_plugin = pytest.mark.skipif(
    not (HAS_ONNX and HAS_EDGELLM_PLUGIN),
    reason="requires tensorrt_edgellm and EDGELLM_PLUGIN_PATH (libNvInfer_edgellm_plugin.so)",
)
requires_qnn = pytest.mark.skipif(
    not HAS_QNN, reason="requires the Qualcomm QNN SDK (QNN_SDK_ROOT)"
)
requires_custom_sdpa = pytest.mark.skipif(
    not HAS_EXECUTORCH_CUSTOM_SDPA,
    reason="requires ExecuTorch's llm custom_ops extension (torch.ops.llama.custom_sdpa/update_cache)",
)
requires_hub = pytest.mark.skipif(
    HF_OFFLINE, reason="requires Hugging Face Hub access (offline mode set)"
)


def summary() -> str:
    return (
        f"cuda={HAS_CUDA} executorch={HAS_EXECUTORCH} "
        f"tensorrt={HAS_TENSORRT} onnx={HAS_ONNX} "
        f"edgellm_plugin={HAS_EDGELLM_PLUGIN} qnn={HAS_QNN} "
        f"executorch_custom_sdpa={HAS_EXECUTORCH_CUSTOM_SDPA} hf_offline={HF_OFFLINE}"
    )
