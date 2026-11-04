import torch
from typing import Union
from .mask_generation import build_unstructured_mask

def _score(weight: torch.Tensor, input_sq_mean: torch.Tensor = None) -> torch.Tensor:
    if input_sq_mean is None:
        raise ValueError("Input activation statistics are required for wanda pruning but are not available.")
    # |W_ij| * ||X_j||_2: the norm over the calibration tokens only rescales every column by the same token count.
    return torch.abs(weight) * torch.sqrt(input_sq_mean.to(weight.device, torch.float32)).unsqueeze(0)

def mask_wanda(
    weight: torch.Tensor,
    input_sq_mean: torch.Tensor,
    pruning_ratio: float = 0.0,
    granularity: Union[str, int] = "neuron",
) -> torch.Tensor:
    scores = _score(weight, input_sq_mean)
    mask = build_unstructured_mask(scores, pruning_ratio, granularity)
    return mask

__all__ = ["mask_wanda"]
