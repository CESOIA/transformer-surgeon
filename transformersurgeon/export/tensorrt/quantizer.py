"""
A minimal, linear-only PT2E quantizer for the TensorRT backend.

TensorRT ingests an ExportedProgram that already carries standard
``quantized_decomposed`` Q/DQ ops; torch-tensorrt's Dynamo converters fold those
into INT8/INT4 layers.  We therefore drive the *same* PT2E flow the ExecuTorch
backends use (prepare_pt2e → calibrate → inject exact surgeon scales →
convert_pt2e), but with a quantizer built purely from ``torchao`` primitives so
this backend has no ExecuTorch dependency.

Only the linear layers named in ``layer_info`` are annotated (shared
``linear_quantizer.LinearOnlyQuantizer``); every other op -- and every
unlisted linear -- stays float, which is what makes mixed-precision export
work.  The observers produced here duck-type as PT2E observers
(``calculate_qparams``), so the shared
``common.inject_scales_into_pt2e_observers`` overrides them with the exact
scales computed by the transformer-surgeon calibration manager.
"""

import warnings
from typing import Any, Optional

import torch

from torchao.quantization.pt2e.observer import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import QuantizationConfig, QuantizationSpec

from ..common import _PRECISION_TO_QRANGE
from ..linear_quantizer import LinearOnlyQuantizer


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


def build_tensorrt_quantizer(layer_info: dict[str, dict[str, Any]]) -> LinearOnlyQuantizer:
    """Build a per-module LinearOnlyQuantizer from compression metadata.

    Each layer in ``layer_info`` gets its own qconfig; layers absent from it are
    not quantised (they remain float).  Layers with stored activation
    calibration use static quantization; others are weight-only.
    """
    quantizer = LinearOnlyQuantizer()
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
