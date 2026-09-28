import torch
from typing import Callable, Tuple, Union

from .vanilla import _dequantize_symmetric, _dequantize_grouped


def quantize_hqq(
    weight: torch.Tensor,
    precision: Union[str, int],
    eps: float = 1e-6,
    granularity: str = "per_channel",
    group_size: int | None = None,
    iters: int = 20,
    early_stop_tol: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
    """Half-quadratic scale-only quantization (symmetric, calibration-free).

    Same algorithm as torchao's ``hqq_scale_only`` (ExecuTorch's LLM recipe):
    start from max-abs scales, then alternate q = clamp(round(W/s)) and the
    least-squares scale s = <W, q> / <q, q> per block until the scales settle.
    Blocks are a whole row (per_channel), groups of ``group_size`` input
    columns (per_group) or the whole matrix (per_tensor). Integer range is the
    full signed range [-2^(b-1), 2^(b-1)-1]. Deterministic.
    """
    if precision == "binary" or not isinstance(precision, int):
        raise ValueError(f"HQQ needs an integer precision, got {precision!r}.")
    W = weight.float()
    out, inp = W.shape
    qmin, qmax = -(2 ** (precision - 1)), 2 ** (precision - 1) - 1
    if granularity == "per_group":
        g = group_size or 128
        if inp % g:
            raise ValueError(f"per_group: in_features {inp} not divisible by group_size {g}.")
        Wg = W.reshape(out, inp // g, g)
    elif granularity == "per_channel":
        Wg = W.reshape(out, 1, inp)
    else:
        Wg = W.reshape(1, 1, -1)
    scale = (Wg.abs().amax(dim=2) / max(abs(qmin), qmax)).clamp_min(eps)
    prev = scale
    for _ in range(max(1, iters)):
        Q = torch.round(Wg / scale.unsqueeze(-1)).clamp(qmin, qmax)
        num = (Wg * Q).sum(dim=2)
        den = (Q * Q).sum(dim=2)
        scale = torch.where(den > 0, num / den, prev).abs().clamp_min(eps)
        if ((scale - prev).abs() / prev).max() < early_stop_tol:
            break
        prev = scale
    Q = torch.round(Wg / scale.unsqueeze(-1)).clamp(qmin, qmax).to(torch.int8)
    if granularity == "per_group":
        return Q.reshape(out, inp), scale, _dequantize_grouped
    if granularity == "per_channel":
        return Q.reshape(out, inp), scale.reshape(out, 1), _dequantize_symmetric
    return Q.reshape(out, inp), scale.reshape(()), _dequantize_symmetric


__all__ = ["quantize_hqq"]
