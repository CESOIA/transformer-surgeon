import importlib.util
import os
import _operator
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx
import torch.nn as nn
from torch.export import export as torch_export

from ..common import (
    ExecutorchExporterConfig,
    ExecuTorchExportResult,
    finalize_export_result,
    resolve_components_and_wrapper,
    extract_layer_quant_info,
    inject_scales_into_pt2e_observers,
    calibrate_pt2e_observers,
)


def is_qnn_available() -> bool:
    """Whether the QNN ExecuTorch backend can actually be used on this machine.

    ``executorch.backends.qualcomm`` unconditionally imports ``cpuinfo`` at
    package-init time and raises a bare ``ImportError`` if it's missing — an
    undeclared transitive dependency, not a real "toolchain not installed"
    signal. There's no side-effect-free way to probe deeper than that (the
    Qualcomm partitioner itself only fails at lowering time), so this checks
    the two cheap preconditions without importing either package: `executorch`
    and `cpuinfo` must both be resolvable, and one of the QNN SDK env vars the
    backend documents must be set. ``importlib.util.find_spec`` only locates a
    module, it does not execute its ``__init__.py``, so this is safe to call
    even when the Qualcomm toolchain isn't installed.
    """
    if importlib.util.find_spec("executorch") is None:
        return False
    if importlib.util.find_spec("cpuinfo") is None:
        return False
    return bool(os.environ.get("QNN_SDK_ROOT") or os.environ.get("QUALCOMM_SDK_ROOT"))


_QNN_UNAVAILABLE_MSG = (
    "QNN backend unavailable: install the Qualcomm ExecuTorch toolchain "
    "(`pip install py-cpuinfo` plus the Qualcomm QNN SDK) and set "
    "QNN_SDK_ROOT (or QUALCOMM_SDK_ROOT) before calling export_with_qnn(). "
    "Use `is_qnn_available()` to check this ahead of time."
)


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


class _BreakQuantAttrsBeforeFp16Linear:
    """
    Edge-dialect pass injected into QnnPassManager's pipeline after LayoutTransform.

    `AnnotateQuantAttrs` sets QCOM_QUANT_ATTRS on activation outputs that feed Q ops.
    `FoldQDQ` then removes those Q/DQ ops, but the annotation remains.  When a
    float16 (un-quantized) linear node follows a quantized layer, its input tensor
    carries QCOM_QUANT_ATTRS.  QNN's define_node reads that attribute to infer a
    quantized dtype (uint8/int8) for the input, making FC(quantized-input, fp16-weight)
    invalid → rejected.

    This pass inserts `edge._to_copy(x, dtype=float16)` before each float16 linear
    whose input has QCOM_QUANT_ATTRS.  The cast node's output is clean float16 (no
    QCOM_QUANT_ATTRS), so QNN sees FC(float16 input, float16 weight) → accepted.
    QNN handles the cast itself as OpCast(uint8 → float16).
    """

    def __init__(self):
        pass

    def __call__(self, graph_module: torch.fx.GraphModule):
        from executorch.backends.qualcomm.builders.node_visitor import dq_ops
        from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
        from executorch.exir.dialects._ops import ops as exir_ops
        from executorch.exir.pass_base import PassResult

        linear_op = exir_ops.edge.aten.linear.default
        _to_copy_op = exir_ops.edge.aten._to_copy.default

        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != linear_op:
                continue

            # Quantized linears: weight comes from a dequantize bypass node.
            # Those have QCOM_QUANT_ATTRS on the weight and QNN handles them via its
            # quantized FC path.  Only float16 linears (plain fp16 weight param) need fixing.
            weight = node.args[1] if len(node.args) > 1 else None
            if isinstance(weight, torch.fx.Node) and weight.target in dq_ops:
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            if not input_node.meta.get(QCOM_QUANT_ATTRS):
                continue

            # Insert _to_copy to produce a clean float16 tensor without QCOM_QUANT_ATTRS.
            with graph.inserting_before(node):
                cast_node = graph.call_function(
                    _to_copy_op,
                    (input_node,),
                    {"dtype": torch.float16},
                )
                cast_node.meta = {
                    k: v for k, v in input_node.meta.items() if k != QCOM_QUANT_ATTRS
                }

            node.replace_input_with(input_node, cast_node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)


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
    if not is_qnn_available():
        raise RuntimeError(_QNN_UNAVAILABLE_MSG)

    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    wrapper, model_config, example_inputs = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    # Detect per-layer compression metadata (hard / soft quantized layers).
    # Quantization is driven entirely by model metadata.
    layer_info = extract_layer_quant_info(wrapper)

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

    if layer_info:
        from executorch.backends.qualcomm.quantizer.quantizer import (
            QnnQuantizer,
            QuantDtype,
            ModuleQConfig,
        )
        from executorch.backends.qualcomm.serialization.qc_schema import (
            QcomChipset,
            QnnExecuTorchBackendType,
        )
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

        # Map torchao precision → QNN QuantDtype.
        # use_8a8w: int8 activations + int8 weights (int8 weight observer uses
        # qmin=-127/qmax=127 symmetric, matching the torchao per-channel scale).
        # use_16a4w: 16-bit activations + 4-bit weights.
        _PREC_TO_QDTYPE = {
            4: QuantDtype.use_16a4w,
            8: QuantDtype.use_8a8w,
        }

        def _make_layer_predicate(layer_names: frozenset) -> Any:
            def _pred(node: torch.fx.Node) -> bool:
                stack = node.meta.get("nn_module_stack", {})
                for _, (qname, _) in reversed(list(stack.items())):
                    if qname in layer_names:
                        return True
                return False
            return _pred

        chipset = _resolve_qcom_chipset(config.soc_model)
        quantizer = QnnQuantizer(
            backend=QnnExecuTorchBackendType.kHtpBackend,
            soc_model=chipset,
        )

        # Group layers by QuantDtype and build submodule_qconfig_list.
        from collections import defaultdict as _defaultdict
        _qdtype_to_layers: dict[QuantDtype, set] = _defaultdict(set)
        for _name, _info in layer_info.items():
            _qdtype = _PREC_TO_QDTYPE.get(_info["precision"])
            if _qdtype is not None:
                _qdtype_to_layers[_qdtype].add(_name)

        _qconfig_list = []
        for _qdtype, _layers in _qdtype_to_layers.items():
            _qconfig = ModuleQConfig(quant_dtype=_qdtype, is_linear_per_channel=True)
            _qconfig_list.append((_make_layer_predicate(frozenset(_layers)), _qconfig))

        # Layers not matched by any predicate stay at the default_quant_config.
        # Set the default to a sentinel that has empty use_per_channel_weight_quant_ops
        # so _get_quant_config returns None (no annotation) for unmatched nodes.
        _float_config = ModuleQConfig()
        _float_config.use_per_channel_weight_quant_ops = {}
        _float_config.quant_config = None
        quantizer.default_quant_config = _float_config

        quantizer.set_submodule_qconfig_list(_qconfig_list)

        prepared = prepare_pt2e(sanitized_module, quantizer)

        calibrate_pt2e_observers(prepared, model_config, config, example_inputs=example_inputs)
        inject_scales_into_pt2e_observers(prepared, layer_info)

        converted = convert_pt2e(prepared)
        with torch.inference_mode():
            quantized_exported = torch_export(
                converted,
                example_inputs,
            )
        model_for_edge = converted
        exported_for_mismatch = quantized_exported

    try:
        from executorch.backends.qualcomm.utils.utils import (
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
            to_edge_transform_and_lower_to_qnn,
        )
    except ImportError as e:
        # is_qnn_available() only checks the cheap preconditions (executorch,
        # cpuinfo, an SDK env var) — this is the actual Qualcomm backend import,
        # so surface a clear error if the toolchain still isn't fully usable.
        raise RuntimeError(_QNN_UNAVAILABLE_MSG) from e
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

    # For mixed-precision models: AnnotateQuantAttrs sets QCOM_QUANT_ATTRS on activation
    # tensors that feed Q ops; FoldQDQ then removes those Q/DQ ops but the annotation
    # persists. Float16 linears that follow quantized layers see a "quantized-dtype" input
    # which makes their FC op invalid for QNN. Inject an edge-dialect cast pass (after
    # LayoutTransform, before TagQuantIO) to insert _to_copy(x, float16) in front of each
    # affected float16 linear, giving QNN a clean float16 input.
    if layer_info:
        from executorch.backends.qualcomm._passes import LayoutTransform, TagQuantIO
        from executorch.backends.qualcomm._passes.qnn_pass_manager import get_capture_program_passes
        from executorch.backends.qualcomm._passes.utils import get_passes_dependency_for_capture_program
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_PASS_ACTIVATE_KEY,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
        )

        passes_job = edge_kwargs.get("passes_job", get_capture_program_passes())
        dep_table = edge_kwargs.get("dep_table", get_passes_dependency_for_capture_program())

        passes_job[_BreakQuantAttrsBeforeFp16Linear] = {
            QCOM_PASS_ACTIVATE_KEY: True,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {},
        }
        # Run after LayoutTransform (which itself runs after FoldQDQ + AnnotateQuantAttrs).
        dep_table[_BreakQuantAttrsBeforeFp16Linear] = [LayoutTransform]
        # Ensure TagQuantIO (and thus ResolveDebugHandle) runs after our pass.
        dep_table[TagQuantIO] = list(dep_table.get(TagQuantIO, [LayoutTransform])) + [
            _BreakQuantAttrsBeforeFp16Linear
        ]

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

    result_precision = "mixed" if layer_info else "full"

    return finalize_export_result(
        pte_path=config.output_path,
        backend="qnn",
        precision=result_precision,
        wrapper=wrapper,
        example_inputs=example_inputs,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=config.run_weight_mismatch_check,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
    )
