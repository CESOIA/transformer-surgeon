import os
import _operator
from dataclasses import dataclass
from typing import Any

import torch
from torch.export import export as torch_export

from ..common import (
    ExecutorchExporterConfig,
    ExecuTorchExportResult,
    finalize_export_result,
    resolve_components_and_wrapper,
)

# Backend-supported quantization configs (subset of common.SUPPORTED_QUANT_CONFIGS).
# "a8w8" / "a8w4" will be added when activation quantization is implemented.
QNN_SUPPORTED_QUANT_CONFIGS: dict[str, str] = {
    "full": "No quantization (FP32)",
    "w8":   "8-bit weight-only (QuantDtype.use_16a8w, fp16 activations, per-channel linear weights)",
    "w4":   "4-bit weight-only (QuantDtype.use_16a4w, fp16 activations, per-channel linear weights)",
}


def _is_wrap_with_set_grad_enabled(target: Any) -> bool:
    target_name = getattr(target, "__name__", "")
    if target_name == "wrap_with_set_grad_enabled":
        return True
    if "WrapWithSetGradEnabled" in str(type(target)):
        return True
    return str(target) == "wrap_with_set_grad_enabled"


def _decompose_wrap_with_set_grad_enabled(graph_module: torch.fx.GraphModule) -> int:
    from executorch.backends.qualcomm._passes.utils import merge_decomposed_graph

    graph = graph_module.graph
    rewritten = 0

    for node in list(graph.nodes):
        if node.op != "call_function" or not _is_wrap_with_set_grad_enabled(node.target):
            continue

        submod_arg = None
        submod_node = None
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                target_attr = str(getattr(arg, "target", ""))
                if "submod" in target_attr:
                    submod_node = arg
                    submod_arg = getattr(graph_module, arg.target)
                    break

        if submod_node is None or submod_arg is None:
            continue

        with graph.inserting_before(node):
            # Skip bool(set_grad_enabled) and submodule arg; remaining args map to placeholders.
            remap = {
                f_arg.name: f_arg for f_arg in node.args[2:] if isinstance(f_arg, torch.fx.Node)
            }

            def _replace_output(
                wwsg_node: torch.fx.Node,
                output_node: torch.fx.Node,
                remap_map: dict,
            ) -> None:
                for user in wwsg_node.users.copy():
                    output_idx = 0
                    is_getitem = False
                    if user.target == _operator.getitem:
                        output_idx = user.args[1]
                        is_getitem = True

                    user.replace_input_with(
                        wwsg_node,
                        remap_map[output_node.args[0][output_idx]],
                    )

                    if is_getitem:
                        for user_user in user.users.copy():
                            user_user.replace_input_with(user, user.args[0])

            merge_decomposed_graph(
                remap=remap,
                target_node=node,
                target_graph=graph,
                decomposed_graph_module=submod_arg,
                output_processor=_replace_output,
            )

        graph.erase_node(node)
        if len(submod_node.users) == 0:
            graph.erase_node(submod_node)
        rewritten += 1

    if rewritten > 0:
        graph.eliminate_dead_code()
        graph_module.recompile()

    return rewritten


def _canonicalize_pow_tensor_scalar(graph_module: torch.fx.GraphModule) -> int:
    from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix

    graph = graph_module.graph
    rewritten = 0

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.pow.Tensor_Scalar:
            continue

        if len(node.args) < 2 or not isinstance(node.args[1], (int, float)):
            continue

        scalar = node.args[1]
        dtype = node.meta["val"].dtype if "val" in node.meta else torch.float32
        device = node.meta["val"].device if "val" in node.meta else torch.device("cpu")
        tensor_value = torch.tensor(float(scalar), dtype=dtype, device=device)

        buffer_name = get_new_attr_name_with_prefix("_pow_scalar_")(graph_module)
        graph_module.register_buffer(buffer_name, tensor_value)

        with graph.inserting_before(node):
            scalar_node = graph.get_attr(buffer_name)
            if "val" in node.meta and hasattr(node.meta["val"], "fake_mode"):
                fake_mode = node.meta["val"].fake_mode
                scalar_node.meta["val"] = fake_mode.from_tensor(tensor_value)

        node.target = torch.ops.aten.pow.Tensor_Tensor
        node.args = (node.args[0], scalar_node)
        rewritten += 1

    if rewritten > 0:
        graph.eliminate_dead_code()
        graph_module.recompile()

    return rewritten


@dataclass
class QNNExportConfig(ExecutorchExporterConfig):
    soc_model: str = "SM8650"
    is_online_prepare: bool = False
    use_fp16: bool = True
    num_shards: int = 1


def _resolve_qcom_chipset(soc_model: Any):
    try:
        from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import QcomChipset
    except ModuleNotFoundError:
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

    if isinstance(soc_model, QcomChipset):
        return soc_model

    if isinstance(soc_model, str):
        if hasattr(QcomChipset, soc_model):
            return getattr(QcomChipset, soc_model)
        normalized = soc_model.upper()
        if hasattr(QcomChipset, normalized):
            return getattr(QcomChipset, normalized)

        supported = sorted(member.name for member in QcomChipset)
        supported_str = ", ".join(supported)
        raise ValueError(
            f"Unsupported soc_model '{soc_model}'. Supported values: {supported_str}"
        )

    raise TypeError(
        "soc_model must be a QcomChipset value or chipset string (for example 'SM8650')."
    )


def export_with_qnn(
    model_or_graph: Any,
    *,
    config: QNNExportConfig,
) -> ExecuTorchExportResult:
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    wrapper, _, example_inputs, quant_plan = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    with torch.inference_mode():
        exported = torch_export(
            wrapper,
            example_inputs,
        )

    sanitized_module = exported.module()
    _decompose_wrap_with_set_grad_enabled(sanitized_module)
    _canonicalize_pow_tensor_scalar(sanitized_module)

    model_for_edge = sanitized_module
    exported_for_mismatch = None

    if quant_plan.global_precision != "full":
        if quant_plan.global_precision not in QNN_SUPPORTED_QUANT_CONFIGS:
            supported = ", ".join(f"'{k}'" for k in QNN_SUPPORTED_QUANT_CONFIGS)
            raise ValueError(
                f"QNN exporter supports: {supported}. "
                f"Got '{quant_plan.global_precision}'."
            )

        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

        _quant_dtype = {
            "w8": QuantDtype.use_16a8w,
            "w4": QuantDtype.use_16a4w,
        }[quant_plan.global_precision]
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            _quant_dtype,
            is_linear_per_channel=True,
        )
        prepared = prepare_pt2e(sanitized_module, quantizer)

        with torch.no_grad():
            prepared(*example_inputs)

        converted = convert_pt2e(prepared)
        with torch.inference_mode():
            quantized_exported = torch_export(
                converted,
                example_inputs,
            )
        model_for_edge = converted
        exported_for_mismatch = quantized_exported

    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    backend_options = generate_htp_compiler_spec(
        use_fp16=config.use_fp16,
        use_multi_contexts=(config.num_shards > 1),
    )
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=_resolve_qcom_chipset(config.soc_model),
        backend_options=backend_options,
        online_prepare=config.is_online_prepare,
    )

    edge_kwargs: dict = {}
    if config.num_shards > 1:
        # llama.fallback.default marks shard boundaries — must be excluded from
        # QNN partitions so the partitioner treats it as a natural split point.
        edge_kwargs["skip_node_op_set"] = {"llama.fallback.default"}

        import re as _re
        from executorch.backends.qualcomm._passes.qnn_pass_manager import get_capture_program_passes
        from executorch.backends.qualcomm._passes.utils import get_passes_dependency_for_capture_program
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_PASS_ACTIVATE_KEY,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
            QCOM_QUANT_ATTRS,
        )
        from executorch.exir.dialects._ops import ops as exir_ops
        from executorch.exir.pass_base import ExportPass, PassResult
        from executorch.extension.llm.custom_ops import model_sharding  # noqa: F401 — registers llama.fallback op

        try:
            n_layers = len(wrapper.decoder.blocks)
        except AttributeError:
            raise RuntimeError(
                "num_shards > 1 requires wrapper.decoder.blocks (TransformerDecoder with a 'blocks' ModuleList)"
            )

        shard_size = n_layers // config.num_shards
        shard_layers = list(range(0, n_layers, shard_size))
        print(f"  Graph sharding: {config.num_shards} shards, layer boundaries at {shard_layers}")

        _pattern = _re.compile(r"blocks\.(\d+)")
        _shard_layers = shard_layers
        _QCOM_QUANT_ATTRS = QCOM_QUANT_ATTRS

        class SplitGraphBlocks(ExportPass):
            def __init__(self, shard_layers):
                super().__init__()
                self.shard_layers = shard_layers

            def _insert_fallback_op(self, graph_module):
                prev_node = None
                prev_layer = None
                for node in graph_module.graph.nodes:
                    if node.op != "call_function" or "nn_module_stack" not in node.meta:
                        continue
                    module_values_list = list(node.meta["nn_module_stack"].values())
                    full_qualified_name = module_values_list[-1][0]
                    match = _pattern.search(full_qualified_name)
                    if match is None:
                        continue
                    cur_layer = int(match.group(1))
                    if cur_layer in self.shard_layers and prev_layer == cur_layer - 1:
                        with graph_module.graph.inserting_after(prev_node):
                            users = list(prev_node.users.keys())
                            inserted_node = graph_module.graph.create_node(
                                "call_function",
                                exir_ops.edge.llama.fallback.default,
                                (prev_node,),
                            )
                            inserted_node.meta["val"] = prev_node.meta["val"]
                            if prev_node.meta.get(_QCOM_QUANT_ATTRS, None):
                                inserted_node.meta[_QCOM_QUANT_ATTRS] = prev_node.meta[_QCOM_QUANT_ATTRS]
                            for user in users:
                                user.replace_input_with(prev_node, inserted_node)
                    prev_layer = cur_layer
                    prev_node = node

            def call(self, graph_module):
                self._insert_fallback_op(graph_module)
                graph_module.recompile()
                return PassResult(graph_module, True)

        passes_job = get_capture_program_passes()
        dep_table = get_passes_dependency_for_capture_program()
        passes_job[SplitGraphBlocks] = {
            QCOM_PASS_ACTIVATE_KEY: True,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {"shard_layers": _shard_layers},
        }
        edge_kwargs["passes_job"] = passes_job
        edge_kwargs["dep_table"] = dep_table

    edge = to_edge_transform_and_lower_to_qnn(
        model_for_edge,
        example_inputs,
        compiler_spec,
        **edge_kwargs,
    )

    executorch_config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=True,
            alloc_graph_output=True,
        ),
    )
    et_program = edge.to_executorch(config=executorch_config)
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
        verbose=config.verbose,
    )
