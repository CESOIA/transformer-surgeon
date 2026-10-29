import torch


def score_gradient(weight: torch.Tensor, weight_grad: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """Per-output-row importance score: L-``norm`` of ``weight * weight_grad``."""
    if weight_grad is None:
        raise ValueError("weight_grad is required for gradient-based scoring but was not provided.")
    return torch.norm(weight * weight_grad, p=norm, dim=1)


__all__ = ["score_gradient"]
