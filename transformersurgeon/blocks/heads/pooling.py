"""Token pooling between an encoder stack and its output head.

Every atom takes the encoder output ``(seq, hidden)`` and returns a rank-2
``(1, features)`` tensor, so the output head always sees a rank-2 input.
"""

import torch


class ClsPooling(torch.nn.Module):
    """First (class) token: ``(1, hidden)``."""

    def forward(self, x):
        return x[:1]


class MeanPooling(torch.nn.Module):
    """Mean over all tokens: ``(1, hidden)``."""

    def forward(self, x):
        return x.mean(dim=0, keepdim=True)


class ClsPatchMeanPooling(torch.nn.Module):
    """Class token concatenated with the mean of the other tokens: ``(1, 2 * hidden)``."""

    def forward(self, x):
        return torch.cat([x[:1], x[1:].mean(dim=0, keepdim=True)], dim=-1)


class NoPooling(torch.nn.Module):
    """Every token (``(seq, hidden)``): for heads applied per token, or none."""

    def forward(self, x):
        return x


__all__ = ["ClsPooling", "MeanPooling", "ClsPatchMeanPooling", "NoPooling"]
