"""
Backend-agnostic, linear-only PT2E quantizer.

Quantization in tsurgeon exports is driven by per-layer compression metadata
(``common.extract_layer_quant_info``): only the linear layers listed there get
Q/DQ, each with its own config, so mixed precision (and float layers left
alone) works. Backends differ only in the QuantizationConfig they build per
layer (XNNPACK: ExecuTorch's symmetric configs, dynamic activations for
weight-only; TensorRT: weight-only or static int8). They share this quantizer,
which annotates nothing but ``aten.linear`` nodes -- in particular not the
add/mul/cat ops of attention and RoPE, whose degenerate calibration would
otherwise clip activations (the reason the stock XNNPACKQuantizer is not used
directly).

Built only from ``torchao.quantization.pt2e`` primitives; importing it does not
require ExecuTorch.
"""

from typing import Any, Callable, Optional

import torch
import torch.fx

from torchao.quantization.pt2e.quantizer import (
    Quantizer,
    annotate_input_qspec_map,
    annotate_output_qspec,
    get_bias_qspec,
    get_input_act_qspec,
    get_module_name_filter,
    get_output_act_qspec,
    get_weight_qspec,
)
from torchao.quantization.pt2e.quantizer.utils import Q_ANNOTATION_KEY


def _is_annotated(node: torch.fx.Node) -> bool:
    annotation = node.meta.get(Q_ANNOTATION_KEY, None)
    return annotation is not None and annotation._annotated


def _mark_annotated(nodes: list[torch.fx.Node]) -> None:
    for node in nodes:
        annotation = node.meta.get(Q_ANNOTATION_KEY, None)
        if annotation is not None:
            annotation._annotated = True


def annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Any,
    filter_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> None:
    """Annotate ``aten.linear`` nodes (optionally filtered) with ``quantization_config``.

    ``quantization_config`` is any object with the PT2E QuantizationConfig
    fields (input_activation, output_activation, weight, bias), e.g. torchao's
    or ExecuTorch's.
    """
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if filter_fn is not None and not filter_fn(node):
            continue
        if _is_annotated(node):
            continue

        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = node.args[2] if len(node.args) > 2 else None

        if input_act_qspec is not None:
            annotate_input_qspec_map(node, act_node, input_act_qspec)
        annotate_input_qspec_map(node, weight_node, weight_qspec)
        nodes_to_mark = [node, weight_node]
        if bias_node is not None and bias_qspec is not None:
            annotate_input_qspec_map(node, bias_node, bias_qspec)
            nodes_to_mark.append(bias_node)
        if output_act_qspec is not None:
            annotate_output_qspec(node, output_act_qspec)
        _mark_annotated(nodes_to_mark)


class LinearOnlyQuantizer(Quantizer):
    """PT2E quantizer annotating only the linear layers registered via
    ``set_module_name``, each with its own config; unregistered layers stay float."""

    def __init__(self) -> None:
        super().__init__()
        self._module_qconfigs: dict[str, Any] = {}

    def set_module_name(self, name: str, qconfig: Any) -> "LinearOnlyQuantizer":
        self._module_qconfigs[name] = qconfig
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for module_name, qconfig in self._module_qconfigs.items():
            annotate_linear(model, qconfig, get_module_name_filter(module_name))
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass


__all__ = ["LinearOnlyQuantizer", "annotate_linear"]
