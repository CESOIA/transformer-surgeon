import torch
from .mask_generation import build_structured_mask

def _score(weight: torch.Tensor, weight_grad: torch.Tensor, norm: int = 2) -> torch.Tensor:
    if weight.grad is None:
        raise ValueError("Gradient is required for gradient-based scoring but is not available.")
    return torch.norm(weight * weight_grad, p=norm, dim=1)

def mask_gradient(
    weight: torch.Tensor,
    weight_grad: torch.Tensor,
    norm: int = 2,
    pruning_ratio: float = 0.0,
) -> torch.Tensor:
    scores = _score(weight, weight_grad, norm)
    mask = build_structured_mask(scores, pruning_ratio)
    return mask

__all__ = ["mask_gradient"]
