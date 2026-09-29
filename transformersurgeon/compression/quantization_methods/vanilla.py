import torch
from typing import Callable, Tuple, Union


def _dequantize_symmetric(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # scale is either scalar or shape (out_channels, 1, ...) for per-channel
    return qdata.to(torch.float32) * scale


def _dequantize_grouped(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # scale (out, in // g): repeat each group scale over its g input columns
    g = qdata.shape[-1] // scale.shape[-1]
    return qdata.to(torch.float32) * scale.repeat_interleave(g, dim=-1)


def _quantize_maxabs(
    weight: torch.Tensor,
    precision: int,
    eps: float = 1e-6,
    per_channel: bool = False,
    group_size: int | None = None,
    group_scale_bits: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
    qmax = 2 ** (precision - 1) - 1
    if group_size is not None:
        out, inp = weight.shape
        if inp % group_size:
            raise ValueError(f"per_group: in_features {inp} not divisible by group_size {group_size}.")
        w = weight.float().reshape(out, inp // group_size, group_size)
        scale = w.abs().amax(dim=-1) / qmax
        if group_scale_bits is not None:
            scale = lpbq_group_scales(scale, group_scale_bits)
        q = torch.clamp(torch.round(w / (scale.unsqueeze(-1) + eps)), -qmax, qmax).to(torch.int8)
        return q.reshape(out, inp), scale, _dequantize_grouped
    if per_channel:
        # reduce over all dims except the output-channel (dim 0)
        reduce_dims = tuple(range(1, weight.dim()))
        scale = weight.abs().amax(dim=reduce_dims, keepdim=True) / qmax
    else:
        scale = weight.abs().max() / qmax
    q = torch.clamp(torch.round(weight / (scale + eps)), -qmax, qmax).to(torch.int8)
    return q, scale, _dequantize_symmetric


def lpbq_group_scales(scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Restrict (out, groups) group scales to ``channel_scale * m`` with an
    integer ``m`` in ``[1, 2**bits]`` and ``channel_scale = max group scale /
    2**bits`` -- Qualcomm HTP's low-power block quantization (LPBQ, bits=4),
    which stores exactly this: one float per channel plus a 4-bit multiplier
    per block. ``m`` is rounded up, so no group's range is clipped (the
    largest group keeps its scale exactly)."""
    steps = 2 ** int(bits)
    channel = scale.amax(dim=-1, keepdim=True) / steps
    safe = torch.where(channel > 0, channel, torch.ones_like(channel))
    m = torch.clamp(torch.ceil(scale / safe - 1e-4), 1, steps)
    return torch.where(channel > 0, channel * m, scale)


def _quantize_binarize(
    weight: torch.Tensor,
    per_channel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
    s = torch.sign(weight).to(torch.int8)
    if per_channel:
        reduce_dims = tuple(range(1, weight.dim()))
        scale = weight.abs().mean(dim=reduce_dims, keepdim=True)
    else:
        scale = weight.abs().mean()
    return s, scale, _dequantize_symmetric


def quantize_vanilla(
    weight: torch.Tensor,
    precision: Union[str, int],
    eps: float = 1e-6,
    granularity: str = "per_tensor",
    group_size: int | None = None,
    group_scale_bits: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
    per_channel = granularity == "per_channel"
    if granularity == "per_group":
        if precision == "binary":
            raise NotImplementedError("per_group granularity is not supported for binary precision.")
        return _quantize_maxabs(weight, precision, eps, group_size=group_size or 128,
                                group_scale_bits=group_scale_bits)
    if precision == "binary":
        return _quantize_binarize(weight, per_channel=per_channel)
    return _quantize_maxabs(weight, precision, eps, per_channel=per_channel)


__all__ = ["quantize_vanilla", "lpbq_group_scales", "_dequantize_symmetric"]
