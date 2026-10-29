"""Read the quantisation decided by the compression step from the layers to export.

``compression/quantization.py`` leaves its decision on every quantised linear
layer and describes it with ``export_quant_info``. This module collects one
backend-independent description per layer, which every backend then lowers in
its own way.
"""

from typing import Any

import torch.nn as nn

from ..compression.quantization import export_quant_info

# Symmetric weight quantisation ranges by bit width, as used by the PT2E observers.
_PRECISION_TO_QRANGE: dict[int, dict[str, int]] = {
    4: {"weight_qmin": -8,   "weight_qmax": 7},
    8: {"weight_qmin": -127, "weight_qmax": 127},
}

# Denominator for converting a per-channel torchao scale back to observer min/max,
# chosen to match the PT2E observer's own formula: (weight_qmax - weight_qmin) / 2.
# Injection sets max_abs = scale * denom so the observer reproduces pt2e_scale = scale exactly.
_PRECISION_TO_EFFECTIVE_DENOM: dict[int, float] = {
    4: 7.5,   # (7 - (-8)) / 2
    8: 127.0, # (127 - (-127)) / 2
}


def extract_layer_quant_info(wrapper: nn.Module) -> dict[str, dict[str, Any]]:
    """Scan the wrapper for compression-tagged linear layers.

    Returns a dict  {layer_name: info}  where info contains:

        scale        – per-channel weight scale tensor (None for soft-quant)
        precision    – integer bit-width (4, 8, …)
        hard         – True for torchao hard-quant, False for soft-quant
        per_channel  – True if scale is per output-channel, False for per-tensor
        act_scale    – activation scale (None if not calibrated)
        act_zero_point, act_precision, act_scheme  (only when act_scale is set)

    Side effects:
      - Hard-quantised weights are dequantised in-place so torch.export sees
        plain float tensors.
      - Activation fake-quant hooks are removed (PT2E handles quantisation
        from this point on).

    Layers without any compression tag are not included in the output.
    """
    layer_info: dict[str, dict[str, Any]] = {}
    for name, module in wrapper.named_modules():
        if isinstance(module, nn.Linear):
            info = export_quant_info(module)  # dequantises the weight; removes the hooks
            if info is not None:
                layer_info[name] = info
    return layer_info


__all__ = ["extract_layer_quant_info"]
