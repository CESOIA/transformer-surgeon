"""Post-export sanity checks: weight mismatches and output error statistics."""

from typing import Any

import torch
import torch.nn as nn


def dequant_from_exported_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """Reconstruct dequantized weight tensors from a PT2E exported state dict.

    PT2E stores quantised weights as _frozen_paramN (int8) alongside
    _scale_N and optionally _zero_point_N tensors.
    """
    dequantized = []
    for key, int_w in state_dict.items():
        if not key.startswith("_frozen_param") or int_w.ndim != 2:
            continue

        suffix    = key.replace("_frozen_param", "")
        scale_key = f"_scale_{suffix}"
        if scale_key not in state_dict:
            continue

        scale = state_dict[scale_key].float()
        zp    = state_dict.get(f"_zero_point_{suffix}")
        zp    = zp.float() if zp is not None else torch.zeros_like(scale)

        dequantized.append((int_w.float() - zp.unsqueeze(1)) * scale.unsqueeze(1))

    return dequantized


def find_weight_mismatches(
    wrapper: nn.Module,
    quantized_exported: Any,
    eps: float,
) -> list[dict[str, Any]]:
    """Best-effort match of exported dequantized weights against the wrapper's float weights.

    For each dequantized exported weight, finds the original layer whose shape
    matches and whose absolute error exceeds eps.  Shape-based matching is
    approximate but sufficient for a sanity check.
    """
    # Detect the model's native dtype from regular (non-quantized) float parameters.
    # This determines the precision at which torchao dequantizes INT4 weights.
    compare_dtype = torch.float32
    for _, tensor in wrapper.state_dict().items():
        if tensor.ndim >= 2 and type(tensor) is torch.Tensor and tensor.is_floating_point():
            compare_dtype = tensor.dtype
            break

    original_candidates = [
        (name, tensor.detach().float())
        for name, tensor in wrapper.state_dict().items()
        if tensor.ndim == 2 and tensor.is_floating_point()
    ]

    state_dict_attr = getattr(quantized_exported, "state_dict", None)
    exported_state_dict = state_dict_attr() if callable(state_dict_attr) else state_dict_attr
    if not isinstance(exported_state_dict, dict):
        raise TypeError("exported object does not expose a dict-like state_dict")

    mismatches = []
    for dq in dequant_from_exported_state_dict(exported_state_dict):
        same_shape = [(n, t) for n, t in original_candidates if list(t.shape) == list(dq.shape)]
        if not same_shape:
            continue

        # Compare at the model's native dtype so torchao float16/float32 dequantization
        # and our float32 reconstruction are evaluated at the same precision.
        dq_cmp = dq.to(compare_dtype).float()
        best_name, best_max, best_mean = None, None, None
        for name, t in same_shape:
            err      = (t.to(compare_dtype).float() - dq_cmp).abs()
            max_err  = float(err.max().item())
            mean_err = float(err.mean().item())
            if best_max is None or max_err < best_max:
                best_name, best_max, best_mean = name, max_err, mean_err

        # Use mean_abs_err to detect systematic scale injection failures.
        # max_abs_err catches at most a few boundary elements off by 1 quant step
        # (e.g. torchao INT8 uses qmin=-128 while backends use qmin=-127), which
        # results in mean≈0 even when max≈1 step.  A real scale error affects
        # most elements and produces a large mean.
        if best_mean is not None and best_mean > eps:
            mismatches.append({
                "matched_weight": best_name,
                "max_abs_err":    best_max,
                "mean_abs_err":   best_mean,
                "eps":            eps,
            })

    return mismatches


def reference_logits(wrapper: nn.Module, inputs: tuple[Any, ...]) -> torch.Tensor:
    """Float-wrapper logits for ``inputs`` (the first output in io_* cache modes)."""
    wrapper.eval()
    with torch.no_grad():
        y = wrapper(*inputs)
    return y[0] if isinstance(y, (tuple, list)) else y


def logit_error_stats(y_ref: torch.Tensor, y_out: Any) -> dict[str, float]:
    """max/mean absolute error, MSE and RMSE of a backend's logits vs the reference."""
    if isinstance(y_out, (tuple, list)):
        y_out = y_out[0]
    if not isinstance(y_out, torch.Tensor):
        y_out = torch.tensor(y_out)
    y_out = y_out.to(y_ref.device, y_ref.dtype)
    err = (y_ref - y_out).abs()
    mse = ((y_ref - y_out) ** 2).mean()
    return {
        "max_abs_err":  float(err.max().item()),
        "mean_abs_err": float(err.mean().item()),
        "mse":          float(mse.item()),
        "rmse":         float(torch.sqrt(mse).item()),
    }


__all__ = ["dequant_from_exported_state_dict", "find_weight_mismatches", "reference_logits", "logit_error_stats"]
