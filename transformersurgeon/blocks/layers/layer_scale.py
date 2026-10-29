"""Per-channel scaling of a residual branch (layer scale, as in CaiT and DINOv2)."""

import torch


class LayerScale(torch.nn.Module):
    """Multiplies the last dimension by a learned vector. The parameter is named ``lambda1``
    like the Hugging Face layer-scale modules, so their state dict loads unchanged."""

    def __init__(self, dim, dtype=None):
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        return x * self.lambda1


__all__ = ["LayerScale"]
