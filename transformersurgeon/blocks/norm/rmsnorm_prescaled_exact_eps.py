"""Max-abs prescaled RMS normalisation that adds eps exactly as the unscaled formulation does."""

import torch

from .rmsnorm import RMSNorm


class RMSNormPrescaledExactEps(RMSNorm):
    """Divides each row by ``m = max(|x|, 1)`` as :class:`RMSNormPrescaled`, but adds ``eps / m**2`` to the
    variance instead of ``eps``, so ``(x/m) * rsqrt(mean((x/m)**2) + eps/m**2)`` equals the unscaled norm exactly.
    Each squared term stays <= 1, so the reduction is float16-safe even with float16 accumulation.
    ``eps / m**2`` underflows only where it is negligible against the variance (>= 1/n)."""

    def forward(self, x):
        max_x = torch.clamp(torch.max(x.abs(), dim=-1, keepdim=True).values, min=1.0)
        x = x / max_x
        eps = (self.eps / max_x) / max_x
        variance = (x * x).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + eps))


__all__ = ["RMSNormPrescaledExactEps"]
