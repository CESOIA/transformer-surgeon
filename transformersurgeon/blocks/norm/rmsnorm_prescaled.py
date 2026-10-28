"""Root-mean-square normalisation with a max-absolute-value rescale first."""

import torch

from .rmsnorm import RMSNorm, rms_normalize


class RMSNormPrescaled(RMSNorm):
    """Divides each row by ``max(|x|, 1)`` before the plain RMS normalisation.

    The rescale keeps ``sum(x^2)`` away from float16 overflow on wide hidden sizes.
    It costs an absolute value, a maximum, a clamp and a division in front of every
    normalisation, and prevents backends from fusing the normalisation into one
    operation. The result is not bit-identical to :class:`RMSNorm`, because the
    rescale changes how ``eps`` enters the variance.
    """

    def forward(self, x):
        max_x = torch.max(x.abs(), dim=-1, keepdim=True).values
        x = x / torch.clamp(max_x, min=1.0)
        return self.weight * rms_normalize(x, self.eps)


__all__ = ["RMSNormPrescaled"]
