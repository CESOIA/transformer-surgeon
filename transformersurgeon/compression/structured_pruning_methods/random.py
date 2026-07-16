import torch
from .mask_generation import build_structured_mask


def score_random(weight: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """Per-output-row random importance score."""
    return torch.rand(weight.size(0), device=weight.device)


def mask_random(
    weight: torch.Tensor,
    norm: int = 2,
    pruning_ratio: float = 0.0,
    granularity="layer",
) -> torch.Tensor:
    scores = score_random(weight, norm=norm)
    return build_structured_mask(scores, pruning_ratio, granularity)


__all__ = ["score_random", "mask_random"]
