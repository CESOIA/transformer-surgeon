"""Read the quantisation decided by the compression step from the layers to export.

``compression/quantization.py`` leaves tags on every quantised linear layer (see
``QUANT_RUNTIME_BUFFERS``/``QUANT_RUNTIME_ATTRS`` there). This module turns those
tags into one backend-independent description per layer, which every backend
then lowers in its own way.
"""

import warnings
from typing import Any

import torch
import torch.nn as nn

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

# Human-readable precision identifiers shared across backends and CLI args.
# weight_bits / activation_bits = None means that axis is not quantised.


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
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        AffineQuantizedTensor = None

    layer_info: dict[str, dict[str, Any]] = {}

    for name, module in wrapper.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        info = _extract_weight_quant_info(module, AffineQuantizedTensor)
        if info is None:
            continue

        _extract_act_quant_info(module, info)  # adds act_* keys; removes hook
        layer_info[name] = info

    return layer_info


def _extract_weight_quant_info(
    module: nn.Linear,
    AffineQuantizedTensor: type | None,
) -> dict[str, Any] | None:
    """Return weight quantization info for one linear layer, or None if unquantised."""

    has_torchao_tag = hasattr(module, '_torchao_precision')
    is_live_aqt = (
        AffineQuantizedTensor is not None
        and isinstance(module.weight, AffineQuantizedTensor)
    )
    has_stashed_scale = hasattr(module, '_torchao_scale')

    if has_torchao_tag and (is_live_aqt or has_stashed_scale):
        # Hard-quantised layer.
        #
        # LRD factors are quantized separately (quantization runs after LRD):
        # this module's own weight is the U factor [out, rank] and its stashed
        # scale has one row per output row; linear_V is visited as its own
        # nn.Linear. A stash whose rows do not match the weight belongs to a
        # pre-LRD weight -- refuse instead of exporting the layer float.
        w_rows = module.weight.shape[0]
        if has_stashed_scale and module._torchao_scale.dim() == 2 and module._torchao_scale.shape[0] not in (1, w_rows):
            raise RuntimeError(
                f"{module}: stashed quantization scale {tuple(module._torchao_scale.shape)} does not match "
                f"weight {tuple(module.weight.shape)} (quantized before LRD?).")

        if has_stashed_scale and not is_live_aqt:
            w = module.weight
            if hasattr(w, "dequantize") and type(w) is not torch.Tensor and type(w) is not nn.Parameter:
                module.weight = nn.Parameter(w.dequantize(), requires_grad=False)
            elif isinstance(w, nn.Parameter) and hasattr(w.data, "dequantize") and type(w.data) is not torch.Tensor:
                module.weight = nn.Parameter(w.data.dequantize(), requires_grad=False)

        if is_live_aqt:
            # Weight is still a torchao AffineQuantizedTensor (direct model pass).
            # Dequantise in-place so torch.export sees a clean float parameter.
            w = module.weight
            scale       = w.tensor_impl.scale.detach().clone()
            per_channel = scale.numel() > 1
            module.weight = nn.Parameter(w.dequantize(), requires_grad=False)
        else:
            # Weight was dequantised earlier by _load_state_dict_dequantized
            # (convert_for_export path); scale was stashed as _torchao_scale.
            scale       = module._torchao_scale.detach().clone()
            per_channel = getattr(module, '_torchao_per_channel', scale.numel() > 1)

        group_size = getattr(module, '_torchao_group_size', None)
        return {
            "scale":       scale,
            "precision":   module._torchao_precision,
            "hard":        True,
            "per_channel": per_channel and group_size is None,
            # per-group: scale is (out_features, in_features // group_size)
            "group_size":  group_size,
            # LPBQ-style group scales (channel scale x integer in [1, 2**bits]), or None
            "group_scale_bits": getattr(module, '_torchao_group_scale_bits', None),
            # int8 asymmetric per-token dynamic activations (XNNPACK 8daXw)
            "act_dynamic": bool(getattr(module, '_torchao_act_dynamic', False)),
        }

    if hasattr(module, '_soft_quant_precision'):
        # Soft-quantised layer: weight is already fake-quantised float.
        # Scales must be re-derived from a calibration pass.
        warnings.warn(
            f"Layer '{module}' was soft-quantized; re-deriving scale via calibration pass.",
            stacklevel=3,
        )
        return {
            "scale":       None,
            "precision":   module._soft_quant_precision,
            "hard":        False,
            "per_channel": True,  # soft-quant always uses per-channel observers
            "group_size":  None,
            "act_dynamic": False,
        }

    return None  # unquantised — skip


def _extract_act_quant_info(module: nn.Linear, info: dict) -> None:
    """Populate activation quantization keys in *info* and remove the fake-quant hooks.

    Extracts both input-side and output-side activation calibration.  If a side
    has no stored calibration its *_scale key is set to None.
    """
    # Input activation (pre-linear).
    if hasattr(module, '_act_quant_scale'):
        info["act_scale"]      = module._act_quant_scale.detach().clone()
        info["act_zero_point"] = module._act_quant_zero_point.detach().clone()
        info["act_precision"]  = module._act_quant_precision
        info["act_scheme"]     = module._act_quant_scheme
        # PT2E inserts its own activation observers; the fake-quant hook is no longer needed.
        if hasattr(module, '_act_quant_hook_handle'):
            module._act_quant_hook_handle.remove()
            del module._act_quant_hook_handle
    else:
        info["act_scale"] = None

    # Output activation (post-linear).
    if hasattr(module, '_act_out_quant_scale'):
        info["act_out_scale"]      = module._act_out_quant_scale.detach().clone()
        info["act_out_zero_point"] = module._act_out_quant_zero_point.detach().clone()
        info["act_out_precision"]  = module._act_out_quant_precision
        info["act_out_scheme"]     = module._act_out_quant_scheme
        if hasattr(module, '_act_out_quant_hook_handle'):
            module._act_out_quant_hook_handle.remove()
            del module._act_out_quant_hook_handle
    else:
        info["act_out_scale"] = None


__all__ = ["extract_layer_quant_info"]
