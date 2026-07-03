"""
ExecuTorch-specific export utilities.

The backend-agnostic machinery (model wrapper, per-layer quant-metadata
extraction, PT2E scale injection / calibration, weight-mismatch checks and the
result container) lives one level up in ``transformersurgeon.export.common`` and
is shared with every backend.  This module re-exports those names for backward
compatibility and adds the parts that genuinely depend on ExecuTorch:

  * ExecuTorchExportResult  – result with a ``pte_path`` alias
  * ExecutorchExporterConfig – exporter config base (adds dynamic_shapes)
  * build_quantizer_from_layer_info / _LinearOnlyQuantizer – XNNPACK PT2E quantizer
  * run_simple_inference_stats – error stats via the ExecuTorch runtime
  * finalize_export_result – ExecuTorch-flavoured wrapper over the generic one
"""

import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx
import torch.nn as nn

# Re-export the backend-agnostic helpers so existing
# `from ...common import <name>` imports keep working.
from ..common import (  # noqa: F401
    SUPPORTED_QUANT_CONFIGS,
    _PRECISION_TO_QRANGE,
    _PRECISION_TO_EFFECTIVE_DENOM,
    BackendExportResult,
    ExporterConfig,
    LLMWrapper,
    build_wrapper,
    build_example_inputs,
    resolve_components_and_wrapper,
    extract_layer_quant_info,
    inject_scales_into_pt2e_observers,
    calibrate_pt2e_observers,
    dequant_from_exported_state_dict,
    find_weight_mismatches,
    finalize_export_result as _finalize_export_result,
)


# ---------------------------------------------------------------------------
# ExecuTorch-flavoured data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecuTorchExportResult(BackendExportResult):
    """Backend result with a ``pte_path`` alias for the on-disk ``.pte`` file."""

    @property
    def pte_path(self) -> str:
        return self.output_path


@dataclass
class ExecutorchExporterConfig(ExporterConfig):
    """Abstract base config shared by all ExecuTorch backend exporters."""


# ---------------------------------------------------------------------------
# XNNPACK PT2E quantizer construction
# ---------------------------------------------------------------------------

class _LinearOnlyQuantizer:
    """Thin wrapper around XNNPACKQuantizer that restricts annotation to
    linear (and linear_relu) patterns only.

    The default XNNPACKQuantizer also annotates cat/add/mul patterns, but
    their annotators ignore the per-module filter_fn (a known limitation of
    the upstream implementation).  This causes attention ops (ROPE sub/add,
    KV-cache cat) to receive observers whose calibration data is degenerate
    (all-negative outputs → zp=127 → clips all positive activations to 0).
    By restricting to linear patterns only we avoid annotating non-linear ops
    in unrelated modules while still enabling INT8 weight+activation fusion
    for the targeted MLP projections.
    """

    def __init__(self, base: "XNNPACKQuantizer"):
        self._base = base
        # Monkey-patch SUPPORTED_PATTERNS to linear-only for the duration of annotate()
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import QuantPattern, LINEAR_TARGETS
        self._linear_only_patterns = [
            QuantPattern("linear", True, False, LINEAR_TARGETS),
            QuantPattern("linear_relu", False, False, LINEAR_TARGETS),
        ]

    def __getattr__(self, name):
        return getattr(self._base, name)

    def annotate(self, model):
        original = self._base.__class__.SUPPORTED_PATTERNS
        self._base.__class__.SUPPORTED_PATTERNS = self._linear_only_patterns
        try:
            result = self._base.annotate(model)
        finally:
            self._base.__class__.SUPPORTED_PATTERNS = original
        return result


def build_quantizer_from_layer_info(layer_info: dict[str, dict[str, Any]]):
    """Build a per-module XNNPACKQuantizer driven by compression metadata.

    Each layer in layer_info gets its own qconfig; layers absent from
    layer_info are not quantised (they remain float).

    Layers with stored activation calibration data use static quantization
    (is_dynamic=False); others use dynamic weight-only (is_dynamic=True).
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    base = XNNPACKQuantizer()
    for layer_name, info in layer_info.items():
        qrange = _PRECISION_TO_QRANGE.get(info["precision"])
        if qrange is None:
            warnings.warn(
                f"Unsupported compression precision {info['precision']} for "
                f"layer '{layer_name}'; skipping — layer will remain float.",
                stacklevel=2,
            )
            continue

        is_dynamic = info["act_scale"] is None  # True → weight-only (fp activations); False → static (calibrated activations)
        qconfig = get_symmetric_quantization_config(
            is_per_channel=info["per_channel"],
            is_dynamic=is_dynamic,
            **qrange,
        )
        base.set_module_name(layer_name, qconfig)

    return _LinearOnlyQuantizer(base)


# ---------------------------------------------------------------------------
# ExecuTorch-runtime inference stats + finalize wrapper
# ---------------------------------------------------------------------------

def run_simple_inference_stats(
    wrapper: nn.Module,
    pte_path: str,
    inference_inputs: tuple[Any, ...],
) -> dict[str, float] | None:
    """Run one forward pass through both the float wrapper and the exported program
    and return absolute/MSE error statistics.  Returns None if the ExecuTorch
    runtime is not installed.
    """
    try:
        from executorch.runtime import Runtime
    except (ImportError, ModuleNotFoundError) as e:
        warnings.warn(
            f"ExecuTorch runtime not available; skipping inference stats. ({e})",
            stacklevel=2,
        )
        return None

    wrapper.eval()
    with torch.no_grad():
        y_ref = wrapper(*inference_inputs)

    # Load and run the exported program.
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method_names = _get_method_names(program)
    if "forward" in method_names:
        method_name = "forward"
    elif method_names:
        method_name = method_names[0]
        warnings.warn(
            f"'forward' not found in exported program; using '{method_name}' instead.",
            stacklevel=2,
        )
    else:
        raise RuntimeError(f"Exported program has no runnable methods. pte_path='{pte_path}'")

    y_et = program.load_method(method_name).execute(list(inference_inputs))[0]
    if not isinstance(y_et, torch.Tensor):
        y_et = torch.tensor(y_et)

    err = (y_ref - y_et).abs()
    mse = ((y_ref - y_et) ** 2).mean()
    return {
        "max_abs_err":  float(err.max().item()),
        "mean_abs_err": float(err.mean().item()),
        "mse":          float(mse.item()),
        "rmse":         float(torch.sqrt(mse).item()),
    }


def _get_method_names(program: Any) -> list[str]:
    attr = getattr(program, "method_names", None)
    if callable(attr):
        return list(attr())
    return list(attr) if attr is not None else []


def finalize_export_result(
    *,
    pte_path: str,
    backend: str,
    precision: str,
    wrapper: nn.Module,
    example_inputs: tuple[Any, ...],
    exported_for_mismatch: Any | None,
    run_weight_mismatch_check: bool,
    weight_mismatch_eps: float,
    verbose: bool,
) -> ExecuTorchExportResult:
    """ExecuTorch-flavoured finalize: runs the generic checks and computes
    inference stats through the ExecuTorch runtime."""
    return _finalize_export_result(
        output_path=pte_path,
        backend=backend,
        precision=precision,
        wrapper=wrapper,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=run_weight_mismatch_check,
        weight_mismatch_eps=weight_mismatch_eps,
        verbose=verbose,
        inference_stats_fn=lambda: run_simple_inference_stats(wrapper, pte_path, example_inputs),
        result_cls=ExecuTorchExportResult,
    )
