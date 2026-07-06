import torch
from .mask_generation import build_structured_mask


def score_magnitude(weight: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """Per-output-row importance score: L-``norm`` of each weight row."""
    return torch.norm(weight, p=norm, dim=1)


def mask_magnitude(
    weight: torch.Tensor,
    norm: int = 2,
    pruning_ratio: float = 0.0,
    granularity="layer",
) -> torch.Tensor:
    scores = score_magnitude(weight, norm=norm)
    return build_structured_mask(scores, pruning_ratio, granularity)


__all__ = ["score_magnitude", "mask_magnitude"]
