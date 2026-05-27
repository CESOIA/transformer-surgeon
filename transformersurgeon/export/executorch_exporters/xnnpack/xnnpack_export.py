import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch.export import export as torch_export

from ..common import (
    BaseExporterConfig,
    ExecuTorchExportResult,
    ExportAdapter,
    finalize_export_result,
    resolve_components_and_wrapper,
    build_quantizer,
)


@dataclass
class XNNPACKExportConfig(BaseExporterConfig):
    precision: str = "int4"
    static_seq_len_1: bool = False
    calibration_data: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    example_input_ids: torch.Tensor | None = None
    example_cache_len_tensor: torch.Tensor | None = None
    max_seq_len: int = 2048
    weight_mismatch_eps: float = 1e-4
    example_inputs: tuple[Any, ...] | None = None
    dynamic_shapes: dict[str, Any] | None = None
    run_weight_mismatch_check: bool = True

    @classmethod
    def from_kwargs(
        cls,
        *,
        output_path: str,
        backend: str,
        adapter: ExportAdapter | None = None,
        **kwargs,
    ) -> "XNNPACKExportConfig":
        recognized = {
            "precision",
            "static_seq_len_1",
            "calibration_data",
            "example_input_ids",
            "example_cache_len_tensor",
            "max_seq_len",
            "convert_options",
            "check_ir_validity",
            "verbose",
            "weight_mismatch_eps",
            "allow_backend_fallback",
            "example_inputs",
            "dynamic_shapes",
            "run_weight_mismatch_check",
        }
        unknown = sorted(set(kwargs.keys()) - recognized)
        if unknown:
            unknown_str = ", ".join(unknown)
            raise TypeError(f"Unsupported xnnpack export options: {unknown_str}")

        return cls(
            output_path=output_path,
            backend=backend,
            adapter=adapter or ExportAdapter(),
            precision=kwargs.get("precision", "int4"),
            static_seq_len_1=kwargs.get("static_seq_len_1", False),
            calibration_data=kwargs.get("calibration_data"),
            example_input_ids=kwargs.get("example_input_ids"),
            example_cache_len_tensor=kwargs.get("example_cache_len_tensor"),
            max_seq_len=kwargs.get("max_seq_len", 2048),
            convert_options=kwargs.get("convert_options") or {"use_sdpa": False},
            check_ir_validity=kwargs.get("check_ir_validity", True),
            verbose=kwargs.get("verbose", False),
            weight_mismatch_eps=kwargs.get("weight_mismatch_eps", 1e-4),
            allow_backend_fallback=kwargs.get("allow_backend_fallback", False),
            example_inputs=kwargs.get("example_inputs"),
            dynamic_shapes=kwargs.get("dynamic_shapes"),
            run_weight_mismatch_check=kwargs.get("run_weight_mismatch_check", True),
        )

def export_with_xnnpack(
    model_or_graph: Any,
    *,
    config: XNNPACKExportConfig,
) -> ExecuTorchExportResult:
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    wrapper, _, example_inputs, dynamic_shapes, quant_plan = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    exported = torch_export(
        wrapper,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    )
    export_for_edge = exported
    exported_for_mismatch = None

    if quant_plan.global_precision != "full":
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

        quantizer = build_quantizer(quant_plan.global_precision)
        prepared = prepare_pt2e(exported.module(), quantizer)

        with torch.no_grad():
            if config.calibration_data:
                if config.adapter.calibration_runner is not None:
                    config.adapter.calibration_runner(prepared, config.calibration_data)
                else:
                    for cal_inputs in config.calibration_data:
                        if isinstance(cal_inputs, (tuple, list)):
                            prepared(*cal_inputs)
                        else:
                            prepared(cal_inputs)
            else:
                prepared(*example_inputs)

        converted = convert_pt2e(prepared)
        quantized_exported = torch_export(
            converted,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        export_for_edge = quantized_exported
        exported_for_mismatch = quantized_exported
    elif config.calibration_data:
        warnings.warn(
            "Calibration data is ignored when precision='full' (no quantization).",
            stacklevel=2,
        )

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
        adapter=config.adapter,
        verbose=config.verbose,
    )
