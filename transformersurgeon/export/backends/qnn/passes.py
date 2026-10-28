"""Graph rewrites applied by the Qualcomm QNN exporter.

Each function works on a ``torch.fx`` graph (or an ExecuTorch edge pass) and
fixes a pattern the Qualcomm lowering does not handle, or prepares the graph for
a quantisation choice. None of them decides what is quantised: that comes from
the compression step.
"""

import _operator
import re
from typing import Any

import torch
import torch.fx
import torch.nn as nn

from ....blocks import MHACausal, TransformerDecoder

# Block index in the module path of a converted stack ("blocks.<i>.attn...").
_BLOCK_INDEX = re.compile(r"blocks\.(\d+)")


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


def _untie_shared_weights(wrapper: nn.Module) -> int:
    """Give every nn.Linear its own weight Parameter when it shares one with another
    Linear/Embedding (ExecuTorch's ConvertLinearToConv2d mishandles shared weights)."""
    seen, untied = set(), 0
    for m in wrapper.modules():
        if isinstance(m, nn.Embedding):
            seen.add(id(m.weight))
    for m in wrapper.modules():
        if isinstance(m, nn.Linear):
            if id(m.weight) in seen:
                m.weight = nn.Parameter(m.weight.detach().clone(), requires_grad=False)
                untied += 1
            seen.add(id(m.weight))
    return untied


def _in_layers(node: torch.fx.Node, layer_names) -> bool:
    """Same match as the quantizer's per-layer predicate: any enclosing module."""
    stack = node.meta.get("nn_module_stack", {})
    return any(qname in layer_names for qname, _ in stack.values())


def _split_float_linear_bias(graph_module: torch.fx.GraphModule, quantized_layers) -> list[str]:
    """linear(x, W, b) -> add(linear(x, W), b) for every biased linear outside
    ``quantized_layers`` (those get the 16a16w default config). See
    QNNExportConfig.split_float_linear_bias. Returns the split nodes' module names."""
    graph = graph_module.graph
    split = []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if len(node.args) < 3 or node.args[2] is None or _in_layers(node, quantized_layers):
            continue
        x, weight, bias = node.args[:3]
        with graph.inserting_before(node):
            linear = graph.call_function(torch.ops.aten.linear.default, (x, weight))
            linear.meta = dict(node.meta)
            add = graph.call_function(torch.ops.aten.add.Tensor, (linear, bias))
            add.meta = dict(node.meta)
        node.replace_all_uses_with(add)
        graph.erase_node(node)
        stack = node.meta.get("nn_module_stack", {})
        split.append(list(stack.values())[-1][0] if stack else node.name)
    if split:
        graph.lint()
        graph_module.recompile()
    return split


def _linear_block_sizes(graph_module: torch.nn.Module, block_layers: dict[str, int]) -> dict[str, tuple]:
    """QnnQuantizer block_size_map (node name -> weight block) for the linear ops of
    LPBQ layers: a (1, group_size) block over the (out, in) weight."""
    sizes, found = {}, set()
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target not in (torch.ops.aten.linear.default,):
            continue
        stack = node.meta.get("nn_module_stack", {})
        if not stack:
            continue
        qname = list(stack.values())[-1][0]
        if qname in block_layers:
            sizes[node.name] = (1, block_layers[qname])
            found.add(qname)
    missing = set(block_layers) - found
    if missing:
        raise RuntimeError(f"QNN export: no linear op found for LPBQ layers {sorted(missing)}.")
    return sizes


def _float_attention_nodes(graph_module) -> list[str]:
    """Nodes kept fp16 inside a quant_scope="all" graph: the attention core of every
    MHACausal (RoPE, KV-cache read/write, scores, mask, softmax, A.V -- everything
    in the attention module except its q/k/v/out linears) and the decoder-level
    mask/RoPE-table slicing. These are rooted in float buffers (RoPE tables,
    mutable KV cache, causal mask) that the QNN quantizer does not observe, which
    makes HTP reject the ops (fp16 input, fixed-point output); and the attention
    scores (|logit| up to ~1e4 plus a -65504 mask) cannot be held by a per-tensor
    16-bit scale anyway. PT2E inserts explicit Q/DQ at the boundaries."""
    names = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        stack = list(node.meta.get("nn_module_stack", {}).values())
        if not stack:
            continue
        types = [str(t[1]) if isinstance(t, tuple) else str(t) for t in stack]
        inner = types[-1]
        in_attn = any(MHACausal.__qualname__ in t for t in types)
        is_linear = "Linear" in inner  # q/k/v/out projections (LinearCompressed / nn.Linear)
        at_decoder_root = inner.endswith(f"{TransformerDecoder.__qualname__}'>") or inner.endswith(
            TransformerDecoder.__qualname__)
        if (in_attn and not is_linear) or at_decoder_root:
            names.append(node.name)
    return names


def make_split_graph_blocks_pass(shard_layers):
    """ExecuTorch edge pass inserting ``llama.fallback`` (ExecuTorch's sharding operator)
    at the first node of every block listed in ``shard_layers``: each shard becomes its
    own QNN context."""
    from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.pass_base import ExportPass, PassResult
    from executorch.extension.llm.custom_ops import model_sharding  # noqa: F401  (registers llama.fallback)

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
                full_qualified_name = list(node.meta["nn_module_stack"].values())[-1][0]
                match = _BLOCK_INDEX.search(full_qualified_name)
                if match is None:
                    continue
                cur_layer = int(match.group(1))
                if cur_layer in self.shard_layers and prev_layer == cur_layer - 1:
                    with graph_module.graph.inserting_after(prev_node):
                        users = list(prev_node.users.keys())
                        inserted_node = graph_module.graph.create_node(
                            "call_function", exir_ops.edge.llama.fallback.default, (prev_node,))
                        inserted_node.meta["val"] = prev_node.meta["val"]
                        if prev_node.meta.get(QCOM_QUANT_ATTRS, None):
                            inserted_node.meta[QCOM_QUANT_ATTRS] = prev_node.meta[QCOM_QUANT_ATTRS]
                        for user in users:
                            user.replace_input_with(prev_node, inserted_node)
                prev_layer = cur_layer
                prev_node = node

        def call(self, graph_module):
            self._insert_fallback_op(graph_module)
            graph_module.recompile()
            return PassResult(graph_module, True)

    return SplitGraphBlocks
