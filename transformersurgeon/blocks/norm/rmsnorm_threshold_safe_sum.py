"""Thresholded RMS normalisation whose variance is a sum of pre-scaled squares."""

import torch

from .rmsnorm_threshold import RMSNormThreshold


class RMSNormThresholdSafeSum(RMSNormThreshold):
    """:class:`RMSNormThreshold` with the variance computed as ``sum((x / sqrt(n))**2)`` instead of
    ``mean(x * x)``: the same value, but each term is <= ``threshold**2 / n``, so the reduction stays in float16
    range even when the backend accumulates it in float16 (``mean(x * x)`` of 1536 terms up to 128**2 overflows).
    Breaks the fused-normalisation pattern of backend compilers."""

    def forward(self, x):
        x = self._rescale(x)
        xs = x * (x.shape[-1] ** -0.5)
        variance = (xs * xs).sum(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


__all__ = ["RMSNormThresholdSafeSum"]
