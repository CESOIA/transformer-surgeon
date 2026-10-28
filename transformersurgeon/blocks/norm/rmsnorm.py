"""Root-mean-square normalisation, plain formulation."""

import torch


def rms_normalize(x, eps):
    """``x / sqrt(mean(x^2) + eps)`` over the last dimension.

    Written as ``x * x`` rather than ``x.pow(2)``: ExecuTorch's Qualcomm backend
    fuses ``mul(w, mul(x, rsqrt(mean(x*x) + eps)))`` into a single RmsNorm operation
    but does not recognise the ``pow`` spelling.
    """
    variance = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


class RMSNorm(torch.nn.Module):
    """Root-mean-square normalisation followed by a learned scale (``weight``).

    This is the cheapest formulation and the one backend compilers recognise as
    a single fused operation. In float16 it can overflow when activations are
    large; see :class:`RMSNormPrescaled` and :class:`RMSNormUpcast`.
    """

    def __init__(self, hidden_size, dtype=None, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return self.weight * rms_normalize(x, self.eps)


__all__ = ["RMSNorm", "rms_normalize"]
