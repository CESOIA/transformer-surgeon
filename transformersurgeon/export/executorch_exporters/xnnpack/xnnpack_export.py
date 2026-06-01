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
    build_quantizer,
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

    wrapper, _, example_inputs, quant_plan = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    exported = torch_export(
        wrapper,
        example_inputs,
    )
    export_for_edge = exported
    exported_for_mismatch = None

    if quant_plan.global_precision != "full":
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

        quantizer = build_quantizer(quant_plan.global_precision)
        prepared = prepare_pt2e(exported.module(), quantizer)

        with torch.no_grad():
            prepared(*example_inputs)

        converted = convert_pt2e(prepared)
        quantized_exported = torch_export(
            converted,
            example_inputs,
        )
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

    return finalize_export_result(
        pte_path=config.output_path,
        backend="xnnpack",
        precision=quant_plan.global_precision,
        wrapper=wrapper,
        example_inputs=example_inputs,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=config.run_weight_mismatch_check,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
    )
