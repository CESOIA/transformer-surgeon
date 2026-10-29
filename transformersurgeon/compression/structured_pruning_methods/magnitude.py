import torch


def score_magnitude(weight: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """Per-output-row importance score: L-``norm`` of each weight row."""
    return torch.norm(weight, p=norm, dim=1)


__all__ = ["score_magnitude"]
