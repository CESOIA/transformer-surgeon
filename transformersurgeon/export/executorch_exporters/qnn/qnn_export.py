import inspect
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
)


@dataclass
class QNNExportConfig(BaseExporterConfig):
    precision: str = "int8"
    static_seq_len_1: bool = True
    calibration_data: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    example_input_ids: torch.Tensor | None = None
    example_cache_len_tensor: torch.Tensor | None = None
    max_seq_len: int = 2048
    weight_mismatch_eps: float = 1e-4
    example_inputs: tuple[Any, ...] | None = None
    dynamic_shapes: dict[str, Any] | None = None
    run_weight_mismatch_check: bool = False
    soc_model: str = "SM8650"
    online_prepare: bool = False
    shared_buffer: bool = False
    optrace: bool = False
    dump_intermediate_outputs: bool = False

    @classmethod
    def from_kwargs(
        cls,
        *,
        output_path: str,
        backend: str,
        adapter: ExportAdapter | None = None,
        **kwargs,
    ) -> "QNNExportConfig":
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
            "soc_model",
            "online_prepare",
            "shared_buffer",
            "optrace",
            "dump_intermediate_outputs",
        }
        unknown = sorted(set(kwargs.keys()) - recognized)
        if unknown:
            unknown_str = ", ".join(unknown)
            raise TypeError(f"Unsupported qnn export options: {unknown_str}")

        return cls(
            output_path=output_path,
            backend=backend,
            adapter=adapter or ExportAdapter(),
            precision=kwargs.get("precision", "int8"),
            static_seq_len_1=kwargs.get("static_seq_len_1", True),
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
            run_weight_mismatch_check=kwargs.get("run_weight_mismatch_check", False),
            soc_model=kwargs.get("soc_model", "SM8650"),
            online_prepare=kwargs.get("online_prepare", False),
            shared_buffer=kwargs.get("shared_buffer", False),
            optrace=kwargs.get("optrace", False),
            dump_intermediate_outputs=kwargs.get("dump_intermediate_outputs", False),
        )


def _resolve_qcom_chipset(soc_model: str):
    try:
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    except ImportError:
        from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
            QcomChipset,
        )

    if not hasattr(QcomChipset, soc_model):
        candidates = [name for name in dir(QcomChipset) if name.isupper()]
        candidates_str = ", ".join(sorted(candidates))
        raise ValueError(
            f"Unsupported soc_model '{soc_model}'. Available values: {candidates_str}"
        )

    return getattr(QcomChipset, soc_model)


def _build_compile_spec(config: QNNExportConfig):
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )

    use_fp16 = config.precision == "full"
    backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
    soc_model = _resolve_qcom_chipset(config.soc_model)

    sig = inspect.signature(generate_qnn_executorch_compiler_spec)
    kwargs = {}
    if "soc_model" in sig.parameters:
        kwargs["soc_model"] = soc_model
    if "backend_options" in sig.parameters:
        kwargs["backend_options"] = backend_options
    if "online_prepare" in sig.parameters:
        kwargs["online_prepare"] = config.online_prepare
    if "is_online_prepare" in sig.parameters:
        kwargs["is_online_prepare"] = config.online_prepare
    if "shared_buffer" in sig.parameters:
        kwargs["shared_buffer"] = config.shared_buffer
    if "optrace" in sig.parameters:
        kwargs["optrace"] = config.optrace
    if "dump_intermediate_outputs" in sig.parameters:
        kwargs["dump_intermediate_outputs"] = config.dump_intermediate_outputs

    return generate_qnn_executorch_compiler_spec(**kwargs)


def _build_qnn_quantizer(precision: str):
    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype

    if precision == "int8":
        quant_dtype = QuantDtype.use_8a8w
    elif precision == "int4":
        # Closest Qualcomm preset to low-bit weight quantization.
        quant_dtype = QuantDtype.use_16a4w
    else:
        raise ValueError(f"Unsupported precision '{precision}'. Use 'full', 'int8', or 'int4'.")

    quantizer = QnnQuantizer()
    quantizer.set_default_quant_config(quant_dtype)
    return quantizer


def _strip_export_assert_nodes(exported_program):
    """Remove PT2E assert nodes that QNN partitioning cannot lower."""
    assert_targets = {
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
    }
    graph = exported_program.graph_module.graph
    removed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in assert_targets:
            graph.erase_node(node)
            removed += 1

    if removed:
        graph.lint()
        exported_program.graph_module.recompile()

    return exported_program


def export_with_qnn(
    model_or_graph: Any,
    *,
    config: QNNExportConfig,
) -> ExecuTorchExportResult:
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    wrapper, _, example_inputs, dynamic_shapes, quant_plan = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    if dynamic_shapes is not None:
        warnings.warn(
            "QNN export enforces static shapes; provided dynamic_shapes will be ignored.",
            stacklevel=2,
        )
    dynamic_shapes = None

    # QNN export is inference-only. Disable autograd while tracing.
    with torch.no_grad():
        exported = torch_export(
            wrapper,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
    export_for_edge = exported
    exported_for_mismatch = None

    if quant_plan.global_precision != "full":
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = _build_qnn_quantizer(quant_plan.global_precision)
        with torch.no_grad():
            prepared = prepare_pt2e(exported.module(), quantizer)

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

    from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    export_for_edge = _strip_export_assert_nodes(export_for_edge)
    export_for_edge = QnnPassManager().transform_for_export_pipeline(export_for_edge)

    compile_spec = _build_compile_spec(config)
    edge = to_edge_transform_and_lower(
        export_for_edge,
        compile_config=EdgeCompileConfig(_check_ir_validity=config.check_ir_validity),
        partitioner=[QnnPartitioner(compile_spec)],
    )

    et_program = edge.to_executorch()
    with open(config.output_path, "wb") as f:
        f.write(et_program.buffer)

    return finalize_export_result(
        pte_path=config.output_path,
        backend="qnn",
        precision=quant_plan.global_precision,
        wrapper=wrapper,
        example_inputs=example_inputs,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=config.run_weight_mismatch_check,
        weight_mismatch_eps=config.weight_mismatch_eps,
        adapter=config.adapter,
        verbose=config.verbose,
    )
