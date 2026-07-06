import torch
from .mask_generation import build_structured_mask


def score_gradient(weight: torch.Tensor, weight_grad: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """Per-output-row importance score: L-``norm`` of ``weight * weight_grad``."""
    if weight_grad is None:
        raise ValueError("weight_grad is required for gradient-based scoring but was not provided.")
    return torch.norm(weight * weight_grad, p=norm, dim=1)


def mask_gradient(
    weight: torch.Tensor,
    weight_grad: torch.Tensor,
    norm: int = 2,
    pruning_ratio: float = 0.0,
    granularity="layer",
) -> torch.Tensor:
    scores = score_gradient(weight, weight_grad, norm)
    return build_structured_mask(scores, pruning_ratio, granularity)


__all__ = ["score_gradient", "mask_gradient"]
