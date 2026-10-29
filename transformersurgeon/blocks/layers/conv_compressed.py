"""Convolution layers that can be pruned and quantised (patch embeddings)."""

import torch.nn as nn


class Conv2dCompressed(nn.Conv2d):
    """``nn.Conv2d`` marked as compressible (structured pruning, quantisation).

    Output channels sit on dimension 0 of ``weight``, as for :class:`LinearCompressed`.
    Low-rank factorisation is intentionally not supported: requesting it fails
    instead of silently producing a wrong layer.
    """

    def __repr__(self):
        return f"Conv2dCompressed({super().__repr__()})"


class Conv3dCompressed(nn.Conv3d):
    """Three-dimensional counterpart of :class:`Conv2dCompressed` (for example a temporal patch embedding)."""

    def __repr__(self):
        return f"Conv3dCompressed({super().__repr__()})"


__all__ = ["Conv2dCompressed", "Conv3dCompressed"]
