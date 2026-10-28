"""
SmoothQuant activation smoothing (Xiao et al., 2023).

Static INT8 activation quantization (TensorRT W8A8) fails on input channels with
massive outliers: one per-tensor scale has to cover them, and the other
channels round to zero. SmoothQuant moves part of each input channel's range
into the weights, exactly (float result unchanged):

    s_j = max|X_j| ** alpha / max|W_:,j| ** (1 - alpha)
    producer output channel j  /= s_j      (RMSNorm weight, or a linear's output row + bias)
    consumer input column  j   *= s_j      (every linear that reads the producer's output)

Which producer feeds which consumers is model truth, declared per family in the
indexing under ``'smoothing'`` (producer path -> consumer paths). A producer must
reach its consumers through operations that commute with a per-channel scale:
a norm directly, or ``up_proj`` through ``act(gate) * up``.

This is a weight transform applied once on the HF ``*_c`` model, before
quantization (whose activation calibration then sees the smoothed ranges). It
leaves weights in place and float, so every later step (quantization, convert,
export) is unchanged. For an LRD consumer the input columns are those of its
V factor; for an LRD producer the output rows are those of its U factor.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from ..utils import normalize_batch
from ..utils import get_submodule, infer_model_device, move_to_device


def _input_weight(linear: nn.Module) -> torch.Tensor:
    """The matrix whose columns read the layer input ([out, in]): V for an LRD layer."""
    v = getattr(linear, "linear_V", None)
    return (v.weight if v is not None else linear.weight)


def _check_float(t: torch.Tensor, what: str) -> None:
    if type(t.data) is not torch.Tensor or not t.dtype.is_floating_point:
        raise RuntimeError(f"SmoothQuant: {what} is already quantized; smooth before quantization.")


def resolve_smoothing_groups(model: nn.Module, indexing: dict, consumers_filter=None) -> list[dict]:
    """[{block, producer, producer_path, consumers: [(path, module)]}] for every block.

    ``consumers_filter(module_path) -> bool`` keeps a group only if it accepts
    *all* its consumers (a producer scale must be compensated in every reader).
    """
    groups = []
    for block_indexing in indexing.values():
        spec = block_indexing.get("smoothing")
        if not spec:
            continue
        cfg = model.config
        if block_indexing.get("config_attr"):
            cfg = getattr(cfg, block_indexing["config_attr"])
        num_blocks = getattr(cfg, block_indexing["num_blocks_attr"])
        template = block_indexing["path_template"]
        for i in range(num_blocks):
            for producer, consumers in spec.items():
                full = [template.format(block_index=i, path=c) for c in consumers]
                if consumers_filter is not None and not all(consumers_filter(f) for f in full):
                    continue
                groups.append({
                    "block": i,
                    "producer_path": producer,
                    "producer": get_submodule(model, template.format(block_index=i, path=producer)),
                    "consumers": [(c, get_submodule(model, template.format(block_index=i, path=c))) for c in consumers],
                })
    return groups


@torch.no_grad()
def collect_input_absmax(model: nn.Module, modules: Iterable[nn.Module], calibration_data,
                         max_batches: Optional[int] = None) -> dict[nn.Module, torch.Tensor]:
    """Per-input-channel max |x| of each module's input over the calibration data (fp32)."""
    stats: dict[nn.Module, torch.Tensor] = {}
    handles = []

    def make_hook(mod):
        def hook(_, inputs):
            x = inputs[0].detach().float().abs().reshape(-1, inputs[0].shape[-1]).amax(0)
            stats[mod] = x if mod not in stats else torch.maximum(stats[mod], x)
        return hook

    for m in set(modules):
        handles.append(m.register_forward_pre_hook(make_hook(m)))
    device = infer_model_device(model)
    try:
        for batch_id, batch in enumerate(calibration_data):
            if max_batches is not None and batch_id >= max_batches:
                break
            args, kwargs, _ = normalize_batch(batch, batch_id=batch_id)
            model(*move_to_device(args, device), **move_to_device(kwargs, device))
    finally:
        for h in handles:
            h.remove()
    return {m: v.cpu() for m, v in stats.items()}


@torch.no_grad()
def apply_smoothquant(model: nn.Module, indexing: dict, calibration_data, alpha: float = 0.5,
                      consumers_filter=None, max_batches: Optional[int] = None,
                      min_scale: float = 1e-5) -> dict:
    """Smooth every indexing-declared (producer -> consumers) group in place.

    Returns a summary {"alpha", "groups": n, "max_input_absmax_before"/"_after" per
    consumer path} for logging. Raises if a weight is already quantized.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"SmoothQuant alpha must be in [0, 1], got {alpha}.")
    groups = resolve_smoothing_groups(model, indexing, consumers_filter)
    if not groups:
        return {"alpha": alpha, "groups": 0}
    # One representative reader per group: all consumers read the same tensor.
    readers = [g["consumers"][0][1] for g in groups]
    absmax = collect_input_absmax(model, readers, calibration_data, max_batches=max_batches)

    summary = {"alpha": alpha, "groups": len(groups), "max_input_absmax_before": {}, "max_input_absmax_after": {}}
    for g in groups:
        x_max = absmax[g["consumers"][0][1]]
        weights = [_input_weight(m) for _, m in g["consumers"]]
        for (path, _), w in zip(g["consumers"], weights):
            _check_float(w, f"block {g['block']} {path}")
            if w.shape[1] != x_max.numel():
                raise RuntimeError(f"SmoothQuant: block {g['block']} {path} reads {w.shape[1]} channels, "
                                   f"calibrated input has {x_max.numel()}.")
        w_max = torch.stack([w.detach().float().abs().amax(0).cpu() for w in weights]).amax(0)
        s = (x_max.clamp_min(min_scale) ** alpha / w_max.clamp_min(min_scale) ** (1 - alpha)).clamp_min(min_scale)
        s = torch.where((x_max > 0) & (w_max > 0), s, torch.ones_like(s))  # dead channels: leave alone

        prod = g["producer"]
        if isinstance(prod, nn.Linear) or hasattr(prod, "linear_V"):
            # Output rows of the producer (U factor for LRD); bias follows its rows.
            _check_float(prod.weight, f"block {g['block']} {g['producer_path']}")
            if prod.weight.shape[0] != s.numel():
                raise RuntimeError(f"SmoothQuant: producer {g['producer_path']} has {prod.weight.shape[0]} "
                                   f"output rows, consumers read {s.numel()} channels.")
            prod.weight.data = (prod.weight.data.float() / s.to(prod.weight.device)[:, None]).to(prod.weight.dtype)
            if getattr(prod, "bias", None) is not None:
                prod.bias.data = (prod.bias.data.float() / s.to(prod.bias.device)).to(prod.bias.dtype)
        else:
            # Normalization layer: per-channel weight (RMSNorm has no bias).
            nw = getattr(prod, "weight", None)
            if nw is None or nw.dim() != 1 or nw.numel() != s.numel():
                raise RuntimeError(f"SmoothQuant: producer {g['producer_path']} is not a per-channel norm "
                                   f"matching {s.numel()} channels.")
            if getattr(prod, "bias", None) is not None:
                prod.bias.data = (prod.bias.data.float() / s.to(prod.bias.device)).to(prod.bias.dtype)
            nw.data = (nw.data.float() / s.to(nw.device)).to(nw.dtype)
        for w in weights:
            w.data = (w.data.float() * s.to(w.device)[None, :]).to(w.dtype)

        key = g["consumers"][0][0]
        summary["max_input_absmax_before"][key] = max(summary["max_input_absmax_before"].get(key, 0.0),
                                                     float(x_max.max()))
        summary["max_input_absmax_after"][key] = max(summary["max_input_absmax_after"].get(key, 0.0),
                                                    float((x_max / s).max()))
    return summary


__all__ = ["apply_smoothquant", "collect_input_absmax", "resolve_smoothing_groups"]
