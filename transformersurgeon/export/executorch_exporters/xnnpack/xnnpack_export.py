import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.export import export as torch_export

from ..common import (
    ExecutorchExporterConfig,
    ExecuTorchExportResult,
    finalize_export_result,
    resolve_components_and_wrapper,
    build_quantizer_from_layer_info,
    extract_layer_quant_info,
    inject_scales_into_pt2e_observers,
    calibrate_pt2e_observers,
)


@dataclass
class XNNPACKExportConfig(ExecutorchExporterConfig):
    pass


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

    et_program = edge.to_executorch()
    with open(config.output_path, "wb") as f:
        f.write(et_program.buffer)

    # Summarize precision from layer metadata for the result object.
    result_precision = "mixed" if layer_info else "full"

    return finalize_export_result(
        pte_path=config.output_path,
        backend="xnnpack",
        precision=result_precision,
        wrapper=wrapper,
        example_inputs=example_inputs,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=config.run_weight_mismatch_check,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
    )
