import torch
from .mask_generation import build_structured_mask

def _score(weight: torch.Tensor, norm: int = 2) -> torch.Tensor:
    return torch.rand(weight.size(0), device=weight.device)

def mask_random(
    weight: torch.Tensor,
    norm: int = 2,
    pruning_ratio: float = 0.0,
) -> torch.Tensor:
    scores = _score(weight, norm=norm)
    mask = build_structured_mask(scores, pruning_ratio)
    return mask

__all__ = ["mask_random"]