import torch
from typing import Union
from .mask_generation import build_unstructured_mask

def _score(weight: torch.Tensor) -> torch.Tensor:
    return torch.abs(weight)

def mask_magnitude(
    weight: torch.Tensor,
    pruning_ratio: float = 0.0,
    granularity: Union[str, int] = "layer",
) -> torch.Tensor:
    scores = _score(weight)
    mask = build_unstructured_mask(scores, pruning_ratio, granularity)
    return mask

__all__ = ["mask_magnitude"]
