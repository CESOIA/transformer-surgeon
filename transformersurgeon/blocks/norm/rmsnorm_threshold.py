"""Root-mean-square normalisation that rescales only the rows whose max |x| exceeds a threshold."""

import torch

from .rmsnorm import RMSNorm, rms_normalize


class RMSNormThreshold(RMSNorm):
    """Rows with ``max|x| > threshold`` are divided by ``max|x| / threshold`` (down to max ``threshold``) before
    the plain RMS normalisation; the other rows are untouched.

    Rows below the threshold are normalised exactly as :class:`RMSNorm` (``eps`` enters unscaled). Rows above it
    have a variance of at least ``threshold**2 / hidden_size``, which makes the rescaled ``eps`` negligible (128 on
    a 1536-wide row: < 1e-7 relative), while ``x * x`` stays <= ``threshold**2``: float16-safe for
    ``threshold <= 128``. ``eps`` stays a constant, so a backend compiler can still fuse the normalisation.
    :class:`RMSNormPrescaled` instead adds ``eps`` to the variance of every rescaled row, where it weighs
    ``max|x|**2`` more.

    ``threshold`` comes from the export option ``rmsnorm_threshold`` (``CONFIG_OPTIONS``).
    """

    # constructor argument -> stack-configuration attribute (filled from convert_options)
    CONFIG_OPTIONS = {"threshold": "rmsnorm_threshold"}

    def __init__(self, hidden_size, dtype=None, eps=1e-5, threshold=128.0):
        super().__init__(hidden_size, dtype=dtype, eps=eps)
        self.threshold = float(threshold)

    def _rescale(self, x):
        max_x = torch.max(x.abs(), dim=-1, keepdim=True).values
        return x / torch.clamp(max_x / self.threshold, min=1.0)

    def forward(self, x):
        return self.weight * rms_normalize(self._rescale(x), self.eps)


__all__ = ["RMSNormThreshold"]
