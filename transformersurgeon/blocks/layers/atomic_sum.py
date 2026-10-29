"""Residual addition as a module of its own."""

import torch


class AtomicSum(torch.nn.Module):
    """Adds two tensors. Being a module, the residual join appears by name in exported
    graphs, so exporters and quantisers can target it."""

    def forward(self, x1, x2):
        return x1 + x2


__all__ = ["AtomicSum"]
