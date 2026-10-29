"""Linear layer that can hold a low-rank factorisation, pruning masks and quantisation tags."""

import torch
import torch.nn as nn
from torch.nn import functional as F

from .low_rank import LowRankMixin


class LinearCompressed(LowRankMixin, nn.Linear):
    """Drop-in replacement of ``nn.Linear`` used by every compressible linear layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample. ``0`` means the layer was fully
            pruned: it then returns its input unchanged.
        bias: Whether the layer has an additive bias.
        rank: Rank of the low-rank factorisation, or ``None``/``"full"`` for none.

    Inputs with more than two dimensions are flattened to two dimensions around
    the matrix product and restored afterwards: the same graph then lowers to a
    plain fully-connected operation on every backend (some backends, such as
    Qualcomm QNN, handle rank-2 inputs more robustly).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rank=None, device=None, dtype=None):
        if out_features <= 0:
            # Fully pruned layer: a plain module that returns its input unchanged.
            nn.Module.__init__(self)
            self.skip = True
            self.in_features, self.out_features, self.rank = in_features, 0, "full"
            self.weight = self.bias = self.linear_V = None
            return
        self.skip = False
        super().__init__(in_features=in_features, out_features=out_features, bias=bias,
                         device=device, dtype=dtype)
        self.linear_V = None
        self.init_lrd(rank)

    @property
    def _lrd_rows(self):
        return self.out_features

    @property
    def _lrd_cols(self):
        return self.in_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.skip:
            return x
        restore_shape = None
        if x.dim() > 2:
            restore_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        if isinstance(self.rank, int):
            x = self.linear_V(x)
        y = F.linear(x, self.weight, self.bias)
        if restore_shape is not None:
            y = y.reshape(*restore_shape, y.shape[-1])
        return y

    def __repr__(self):
        return (f"LinearCompressed(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, rank={self.rank})")

    __str__ = __repr__


def linear_from_spec(spec, in_features, out_features, bias=False, dtype=None):
    """Build a :class:`LinearCompressed` from a layer specification.

    ``spec`` is a dict with any of ``in_features``, ``out_features``, ``rank`` and
    ``bias``; missing entries fall back to the given defaults. Specifications are
    produced when a model is converted for export (``export/convert.py``) and
    carry the geometry the source layer already has after compression.
    """
    spec = spec or {}
    return LinearCompressed(
        spec.get("in_features", in_features),
        spec.get("out_features", out_features),
        bias=spec.get("bias", bias),
        rank=spec.get("rank", "full"),
        dtype=dtype,
    )


__all__ = ["LinearCompressed", "linear_from_spec"]
