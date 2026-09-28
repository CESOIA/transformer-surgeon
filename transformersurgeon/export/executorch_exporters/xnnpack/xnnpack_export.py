import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch.export import export as torch_export

from ..common import (
    ExecutorchExporterConfig,
    ExecuTorchExportResult,
    finalize_executorch_result,
    resolve_components_and_wrapper,
    build_quantizer_from_layer_info,
    extract_layer_quant_info,
    inject_scales_into_pt2e_observers,
    calibrate_pt2e_observers,
)


@dataclass
class XNNPACKExportConfig(ExecutorchExporterConfig):
    """quant_path:
      "pt2e"   - (default) PT2E Q/DQ via the linear-only XNNPACK quantizer;
                 per-tensor/per-channel weights only.
      "torchao"- ExecuTorch's LLM recipe: every hard-quantized linear becomes a
                 torchao IntxUnpackedToInt8Tensor built from tsurgeon's exact
                 integer weights and scales (per-channel or per-group), with
                 int8 asymmetric per-token dynamic activations (8da4w / 8da8w),
                 lowered with the dynamically-quantized XNNPACK partitioner
                 first. Supports per-group scales and the final layer (lm_head).
    """
    quant_path: str = "pt2e"


def _torchao_quantize_linears(wrapper, layer_info: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Swap each quantized linear's (already dequantized) weight for an
    IntxUnpackedToInt8Tensor carrying tsurgeon's exact scales. Returns
    {layer_name: granularity} for the layers converted."""
    import torch.nn as nn
    from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
        IntxUnpackedToInt8Tensor,
        IntxUnpackedToInt8TensorActivationQuantization,
    )

    modules = dict(wrapper.named_modules())
    done = {}
    for name, info in layer_info.items():
        if info.get("act_scale") is not None:
            raise NotImplementedError(
                f"XNNPACK torchao path: '{name}' has static activation calibration; only dynamic "
                "per-token int8 activations are supported on this path.")
        if not info.get("hard") or info.get("scale") is None:
            raise NotImplementedError(f"XNNPACK torchao path: '{name}' is not hard-quantized.")
        if not info.get("act_dynamic"):
            warnings.warn(
                f"XNNPACK torchao path: '{name}' is weight-only (no dynamic int8 activations); "
                "XNNPACK has no fp32 x intN weight-only GEMM, so this layer may not be delegated.",
                stacklevel=2)
        bits = int(info["precision"])
        if not 2 <= bits <= 8:
            raise ValueError(f"XNNPACK torchao path: unsupported precision {bits} for '{name}'.")
        mod = modules[name]
        w = mod.weight.detach()
        n, k = w.shape
        scale = info["scale"].detach().to(w.dtype)
        g = info.get("group_size")
        if g is not None:
            scale = scale.reshape(n, k // g)
            block = (1, g)
            gran = f"group{g}"
        elif info.get("per_channel"):
            scale = scale.reshape(n, 1)
            block = (1, k)
            gran = "per_channel"
        else:
            scale = scale.reshape(1, 1).expand(n, 1).contiguous()
            block = (1, k)
            gran = "per_tensor"
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1
        qdata = torch.clamp(
            torch.round(w.float() / scale.float().repeat_interleave(block[1], dim=1)), qmin, qmax
        ).to(torch.int8)
        qw = IntxUnpackedToInt8Tensor(
            qdata=qdata,
            scale=scale,
            zero_point=torch.zeros_like(scale, dtype=torch.int8),
            target_dtype=getattr(torch, f"int{bits}"),
            block_size=block,
            dtype=w.dtype,
            activation_quantization=(IntxUnpackedToInt8TensorActivationQuantization.INT8_ASYM_PER_TOKEN
                                     if info.get("act_dynamic") else None),
        )
        mod.weight = nn.Parameter(qw, requires_grad=False)
        done[name] = gran
    return done


def export_with_xnnpack(
    model_or_graph: Any,
    *,
    config: XNNPACKExportConfig,
) -> ExecuTorchExportResult:
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    wrapper, model_config, example_inputs = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    # Detect per-layer compression metadata (hard / soft quantized layers).
    # Quantization is driven entirely by model metadata — config.precision is not consulted.
    layer_info = extract_layer_quant_info(wrapper)

    if config.quant_path == "torchao":
        return _export_torchao(wrapper, model_config, example_inputs, layer_info, config)
    if config.quant_path != "pt2e":
        raise ValueError(f"Unknown XNNPACK quant_path {config.quant_path!r}.")

    exported = torch_export(
        wrapper,
        example_inputs,
    )
    export_for_edge = exported
    exported_for_mismatch = None

    if layer_info:
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

        quantizer = build_quantizer_from_layer_info(layer_info)
        prepared = prepare_pt2e(exported.module(), quantizer)

        # Calibration pass: run multiple forward passes with real text so that all
        # observers (including output-activation observers on gate/up/down projections)
        # collect representative statistics.  Random-token passes leave output observers
        # with degenerate histograms (gate/up outputs are mostly negative for random
        # tokens → zp=127 → clips all positive activations to 0).
        calibrate_pt2e_observers(prepared, model_config, config, example_inputs=example_inputs)

        # Override observer results for hard-quantized layers with exact surgeon scales.
        inject_scales_into_pt2e_observers(prepared, layer_info)

        converted = convert_pt2e(prepared)
        quantized_exported = torch_export(converted, example_inputs)
        export_for_edge = quantized_exported
        exported_for_mismatch = quantized_exported

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(
        export_for_edge,
        compile_config=EdgeCompileConfig(_check_ir_validity=config.check_ir_validity),
        partitioner=[XnnpackPartitioner()],
    )

    # alloc_graph_input=False keeps graph inputs caller-owned (zero-copy) instead
    # of the MemoryPlanningPass default (alloc_graph_input=True), which copies the
    # input tensor into an internal buffer on every forward() call. ExecuTorch's
    # own llama exporter (extension/llm/export/builder.py) sets this explicitly
    # for the same reason - see docs/investigations/FRAMEWORK_PROBLEMS.md P1.
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    et_program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    with open(config.output_path, "wb") as f:
        f.write(et_program.buffer)

    # Summarize precision from layer metadata for the result object.
    result_precision = "mixed" if layer_info else "full"

    return finalize_executorch_result(
        pte_path=config.output_path,
        backend="xnnpack",
        precision=result_precision,
        wrapper=wrapper,
        example_inputs=example_inputs,
        model_config=model_config,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=config.run_weight_mismatch_check,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
    )


def _export_torchao(wrapper, model_config, example_inputs, layer_info, config: XNNPACKExportConfig):
    """ExecuTorch-LLM-style lowering: torchao 8daXw linears, dynamic-quant partitioner first."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
        XnnpackPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes import MemoryPlanningPass

    converted = _torchao_quantize_linears(wrapper, layer_info) if layer_info else {}
    with torch.no_grad():
        exported = torch_export(wrapper, example_inputs)
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()],
    )
    et_program = edge.to_executorch(
        config=ExecutorchBackendConfig(memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False))
    )
    with open(config.output_path, "wb") as f:
        f.write(et_program.buffer)
    result = finalize_executorch_result(
        pte_path=config.output_path,
        backend="xnnpack",
        precision="mixed" if layer_info else "full",
        wrapper=wrapper,
        example_inputs=example_inputs,
        model_config=model_config,
        exported_for_mismatch=None,
        run_weight_mismatch_check=False,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
    )
    result.quantized_layers = converted
    return result
