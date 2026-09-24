import torch


def score_random(weight: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """Per-output-row random importance score."""
    return torch.rand(weight.size(0), device=weight.device)


__all__ = ["score_random"]
