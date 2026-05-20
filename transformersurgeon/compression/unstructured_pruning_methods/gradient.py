import torch
from typing import Union
from .mask_generation import build_unstructured_mask

def _score(weight: torch.Tensor, weight_grad: torch.Tensor = None) -> torch.Tensor:
    if weight_grad is None:
        raise ValueError("Gradient is required for gradient-based pruning but is not available.")
    return torch.abs(weight * weight_grad)

def mask_gradient(
    weight: torch.Tensor,
    weight_grad: torch.Tensor,
    pruning_ratio: float = 0.0,
    granularity: Union[str, int] = "layer",
) -> torch.Tensor:
    scores = _score(weight, weight_grad)
    mask = build_unstructured_mask(scores, pruning_ratio, granularity)
    return mask

__all__ = ["mask_gradient"]
