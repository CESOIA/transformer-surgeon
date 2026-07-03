"""
TensorRT backend exporter.

Mirrors the ExecuTorch backends' export pipeline but lowers to a TensorRT
engine via ``torch-tensorrt``'s Dynamo path.  All the backend-agnostic
machinery (model wrapper, per-layer quant-metadata extraction, PT2E scale
injection / calibration, weight-mismatch checks, result assembly) is reused
from ``transformersurgeon.export.common``; only the quantizer
(``tensorrt.quantizer``) and the compile/save step are TensorRT-specific.

Mixed quantization works exactly as for the ExecuTorch backends: quantization is
driven entirely by per-layer compression metadata, so a model can mix INT8/INT4
quantized linears with float ones in a single engine.
"""

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.export import export as torch_export

from ..common import (
    ExporterConfig,
    BackendExportResult,
    resolve_components_and_wrapper,
    extract_layer_quant_info,
    inject_scales_into_pt2e_observers,
    calibrate_pt2e_observers,
    finalize_export_result,
)
from .quantizer import build_tensorrt_quantizer


@dataclass
class TensorRTExportResult(BackendExportResult):
    """Backend result with an ``engine_path`` alias for the on-disk artifact."""

    @property
    def engine_path(self) -> str:
        return self.output_path


@dataclass
class TensorRTExportConfig(ExporterConfig):
    """Configuration for the TensorRT exporter.

    use_fp16          - allow FP16 kernels for the un-quantized parts of the graph.
    min_block_size    - smallest number of ops TensorRT will accept as a subgraph.
    workspace_size    - TensorRT builder workspace in bytes (0 = library default).
    truncate_double   - downcast float64 constants to float32 (TensorRT has no f64).
    optimization_level - TensorRT builder optimization level (None = default).
    output_format     - torch-tensorrt save format: "exported_program" | "torchscript".
    device            - target device string for compilation ("cuda:0" by default).
    use_explicit_typing - torch-tensorrt "strong typing" mode. Must stay False for mixed
                          fp16/int8 graphs: newer torch-tensorrt versions require
                          enabled_precisions == {float32} when this is True, which is
                          incompatible with the fp16/int8 mix this exporter builds.
    """

    use_fp16: bool = True
    min_block_size: int = 1
    workspace_size: int = 0
    truncate_double: bool = True
    optimization_level: int | None = None
    output_format: str = "exported_program"
    device: str = "cuda:0"
    use_explicit_typing: bool = False


def _detect_float_dtype(wrapper: nn.Module) -> torch.dtype:
    """Return the model's native floating dtype (first 2-D float parameter)."""
    for _, tensor in wrapper.state_dict().items():
        if tensor.ndim >= 2 and type(tensor) is torch.Tensor and tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def _compile_tensorrt(exported_program: Any, example_inputs: tuple[Any, ...], *, config: "TensorRTExportConfig", enabled_precisions: set):
    """Call torch-tensorrt's Dynamo compile, tolerating the arg_inputs/inputs rename."""
    import torch_tensorrt

    compile_kwargs: dict[str, Any] = {
        "enabled_precisions": enabled_precisions,
        "min_block_size": config.min_block_size,
        "truncate_double": config.truncate_double,
        "device": config.device,
        "use_explicit_typing": config.use_explicit_typing,
    }
    if config.workspace_size:
        compile_kwargs["workspace_size"] = config.workspace_size
    if config.optimization_level is not None:
        compile_kwargs["optimization_level"] = config.optimization_level

    # torch-tensorrt renamed `inputs` → `arg_inputs`; support both.
    try:
        return torch_tensorrt.dynamo.compile(
            exported_program, arg_inputs=list(example_inputs), **compile_kwargs
        )
    except TypeError:
        return torch_tensorrt.dynamo.compile(
            exported_program, inputs=list(example_inputs), **compile_kwargs
        )


def _save_tensorrt(trt_module: Any, output_path: str, example_inputs: tuple[Any, ...], *, output_format: str) -> None:
    import torch_tensorrt

    try:
        torch_tensorrt.save(
            trt_module, output_path, arg_inputs=list(example_inputs), output_format=output_format
        )
    except TypeError:
        torch_tensorrt.save(
            trt_module, output_path, inputs=list(example_inputs), output_format=output_format
        )


def _run_trt_inference_stats(
    wrapper: nn.Module,
    trt_module: Any,
    inference_inputs: tuple[Any, ...],
) -> dict[str, float] | None:
    """Compare a float wrapper forward pass against the compiled TensorRT module.

    Returns None if the TensorRT module cannot be executed here (e.g. no CUDA
    device available), matching the ExecuTorch runtime-absent behaviour.
    """
    def _to_device(x, device):
        # inference_inputs may contain nested KV-cache lists in io_* modes.
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, (list, tuple)):
            return type(x)(_to_device(e, device) for e in x)
        return x

    try:
        wrapper.eval()
        with torch.no_grad():
            y_ref = wrapper(*inference_inputs)
        # io cache modes return (logits, new_key_caches, new_value_caches).
        if isinstance(y_ref, (tuple, list)):
            y_ref = y_ref[0]

        device = next((p.device for p in getattr(trt_module, "parameters", lambda: [])()), None)
        trt_inputs = inference_inputs
        if device is not None and device.type == "cuda":
            trt_inputs = tuple(_to_device(t, device) for t in inference_inputs)

        y_trt = trt_module(*trt_inputs)
        if isinstance(y_trt, (tuple, list)):
            y_trt = y_trt[0]
        y_trt = y_trt.to(y_ref.device, y_ref.dtype)

        err = (y_ref - y_trt).abs()
        mse = ((y_ref - y_trt) ** 2).mean()
        return {
            "max_abs_err":  float(err.max().item()),
            "mean_abs_err": float(err.mean().item()),
            "mse":          float(mse.item()),
            "rmse":         float(torch.sqrt(mse).item()),
        }
    except Exception as exc:  # noqa: BLE001 — stats are best-effort
        import warnings
        warnings.warn(f"TensorRT inference stats skipped: {exc}", stacklevel=2)
        return None


def export_with_tensorrt(
    model_or_graph: Any,
    *,
    config: TensorRTExportConfig,
) -> TensorRTExportResult:
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    wrapper, model_config, example_inputs = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    # Detect per-layer compression metadata (hard / soft quantized layers).
    # Quantization is driven entirely by model metadata — config precision fields
    # only tune the float parts of the graph.
    layer_info = extract_layer_quant_info(wrapper)

    exported = torch_export(wrapper, example_inputs)

    # Base precisions for the un-quantized parts of the graph.
    float_dtype = _detect_float_dtype(wrapper)
    enabled_precisions: set = {torch.float32}
    if config.use_fp16 or float_dtype == torch.float16:
        enabled_precisions.add(torch.float16)

    program_for_trt = exported
    exported_for_mismatch = None

    if layer_info:
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

        quantizer = build_tensorrt_quantizer(layer_info)
        prepared = prepare_pt2e(exported.module(), quantizer)

        # Calibration pass with real text so activation observers collect
        # representative statistics (see common.calibrate_pt2e_observers).
        calibrate_pt2e_observers(prepared, model_config, config, example_inputs=example_inputs)

        # Override observer results with exact surgeon scales (hard weights +
        # calibrated activations).
        inject_scales_into_pt2e_observers(prepared, layer_info)

        converted = convert_pt2e(prepared)
        quantized_exported = torch_export(converted, example_inputs)
        program_for_trt = quantized_exported
        exported_for_mismatch = quantized_exported

        # The quantized linears already carry explicit Q/DQ ops (from convert_pt2e),
        # which is what lets TensorRT build INT8/INT4 layers for them. Do NOT also
        # add torch.int8 to enabled_precisions: that flag asks the builder for
        # *implicit*, network-wide INT8 quantization, which requires a calibrator
        # (or explicit dynamic range) on every tensor in the graph, not just the
        # already-quantized ones — and fails with "Calibration failure occurred
        # with no scaling factors detected" for the untouched float layers.

    trt_module = _compile_tensorrt(
        program_for_trt,
        example_inputs,
        config=config,
        enabled_precisions=enabled_precisions,
    )

    _save_tensorrt(
        trt_module,
        config.output_path,
        example_inputs,
        output_format=config.output_format,
    )

    result_precision = "mixed" if layer_info else "full"

    return finalize_export_result(
        output_path=config.output_path,
        backend="tensorrt",
        precision=result_precision,
        wrapper=wrapper,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=config.run_weight_mismatch_check,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
        inference_stats_fn=lambda: _run_trt_inference_stats(wrapper, trt_module, example_inputs),
        result_cls=TensorRTExportResult,
    )
