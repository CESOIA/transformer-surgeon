"""
A minimal, linear-only PT2E quantizer for the TensorRT backend.

TensorRT ingests an ExportedProgram that already carries standard
``quantized_decomposed`` Q/DQ ops; torch-tensorrt's Dynamo converters fold those
into INT8/INT4 layers.  We therefore drive the *same* PT2E flow the ExecuTorch
backends use (prepare_pt2e → calibrate → inject exact surgeon scales →
convert_pt2e), but with a quantizer built purely from ``torchao`` primitives so
this backend has no ExecuTorch dependency.

Only the linear layers named in ``layer_info`` are annotated; every other op —
and every unlisted linear — stays float, which is what makes mixed-precision
export work.  The observers produced here duck-type as PT2E observers
(``calculate_qparams``), so the shared
``common.inject_scales_into_pt2e_observers`` overrides them with the exact
scales computed by the transformer-surgeon calibration manager.
"""

import warnings
from typing import Any, Callable, Optional

import torch
import torch.fx

from torchao.quantization.pt2e.observer import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationConfig,
    QuantizationSpec,
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

from ..common import _PRECISION_TO_QRANGE


# Weight/activation quantized-storage dtype for PT2E.  Sub-byte precisions (INT4)
# are represented as int8 storage with a narrowed [quant_min, quant_max] range,
# matching the torchao / quantized_decomposed convention.
_QUANT_DTYPE = torch.int8
_EPS = 2 ** -12


def _build_qconfig(precision: int, per_channel: bool, static: bool) -> Optional[QuantizationConfig]:
    """Build a QuantizationConfig for one linear layer.

    static=True  → input (and output) activations are quantized per-tensor int8
                   with a HistogramObserver (calibrated / injected later).
    static=False → weight-only: activations stay float.

    Returns None if the precision is unsupported (caller keeps the layer float).
    """
    qrange = _PRECISION_TO_QRANGE.get(precision)
    if qrange is None:
        return None

    weight_qscheme = (
        torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
    )
    weight_observer = PerChannelMinMaxObserver if per_channel else MinMaxObserver
    weight_spec = QuantizationSpec(
        dtype=_QUANT_DTYPE,
        quant_min=qrange["weight_qmin"],
        quant_max=qrange["weight_qmax"],
        qscheme=weight_qscheme,
        ch_axis=0 if per_channel else None,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer.with_args(eps=_EPS),
    )

    if not static:
        # Weight-only quantization: activations remain float.
        return QuantizationConfig(
            input_activation=None,
            output_activation=None,
            weight=weight_spec,
            bias=None,
            is_qat=False,
        )

    act_spec = QuantizationSpec(
        dtype=_QUANT_DTYPE,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=_EPS),
    )
    return QuantizationConfig(
        input_activation=act_spec,
        output_activation=act_spec,
        weight=weight_spec,
        bias=None,
        is_qat=False,
    )


def _is_annotated(node: torch.fx.Node) -> bool:
    annotation = node.meta.get(Q_ANNOTATION_KEY, None)
    return annotation is not None and annotation._annotated


def _mark_annotated(nodes: list[torch.fx.Node]) -> None:
    for node in nodes:
        annotation = node.meta.get(Q_ANNOTATION_KEY, None)
        if annotation is not None:
            annotation._annotated = True


def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> None:
    """Annotate ``aten.linear`` nodes (optionally filtered by module name) with
    the given qconfig.  Mirrors the reference XNNPACK annotator but uses only
    torchao helpers so no ExecuTorch import is needed."""
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


class TensorRTLinearQuantizer(Quantizer):
    """PT2E quantizer that annotates only the linear layers registered via
    ``set_module_name``.  Each layer carries its own QuantizationConfig, so
    different layers can use different precisions (mixed quantization) and any
    layer left unregistered stays float."""

    def __init__(self) -> None:
        super().__init__()
        self._module_qconfigs: dict[str, QuantizationConfig] = {}

    def set_module_name(self, name: str, qconfig: QuantizationConfig) -> "TensorRTLinearQuantizer":
        self._module_qconfigs[name] = qconfig
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for module_name, qconfig in self._module_qconfigs.items():
            _annotate_linear(model, qconfig, get_module_name_filter(module_name))
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass


def build_tensorrt_quantizer(layer_info: dict[str, dict[str, Any]]) -> TensorRTLinearQuantizer:
    """Build a per-module TensorRTLinearQuantizer from compression metadata.

    Each layer in ``layer_info`` gets its own qconfig; layers absent from it are
    not quantised (they remain float).  Layers with stored activation
    calibration use static quantization; others are weight-only.
    """
    quantizer = TensorRTLinearQuantizer()
    for layer_name, info in layer_info.items():
        static = info["act_scale"] is not None
        qconfig = _build_qconfig(info["precision"], info["per_channel"], static)
        if qconfig is None:
            warnings.warn(
                f"Unsupported compression precision {info['precision']} for "
                f"layer '{layer_name}'; skipping — layer will remain float.",
                stacklevel=2,
            )
            continue
        quantizer.set_module_name(layer_name, qconfig)
    return quantizer
