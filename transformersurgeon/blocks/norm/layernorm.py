"""Layer normalisation."""

import torch


class LayerNorm(torch.nn.LayerNorm):
    """``torch.nn.LayerNorm`` with the constructor signature shared by every normalisation atom."""

    def __init__(self, hidden_size, dtype=None, eps=1e-5):
        super().__init__(hidden_size, eps=eps, dtype=dtype)


__all__ = ["LayerNorm"]
