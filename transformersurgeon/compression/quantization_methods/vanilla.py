import torch
from typing import Union


def _quantize_maxabs(weight: torch.Tensor, precision: int, eps: float=1e-6) -> torch.Tensor:
    qmax = 2**precision - 1
    scale = weight.abs().max() / qmax
    q = torch.clamp(torch.round(weight / (scale + eps)), -qmax, qmax)
    return q * scale


def _quantize_binarize(weight: torch.Tensor) -> torch.Tensor:
    s = torch.sign(weight)
    scale = weight.abs().mean()
    return s * scale


def quantize_vanilla(weight: torch.Tensor, precision: Union[str, int]) -> torch.Tensor:
    if precision == "binary":
        return _quantize_binarize(weight)
    return _quantize_maxabs(weight, precision)

__all__ = ["quantize_vanilla",]
