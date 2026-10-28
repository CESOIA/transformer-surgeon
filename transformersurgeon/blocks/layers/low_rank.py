"""Low-rank factorisation state shared by the compressible layers.

A layer with weight ``W`` of shape ``[rows, cols]`` can be replaced by two factors:
``weight`` of shape ``[rows, rank]`` and ``linear_V.weight`` of shape ``[rank, cols]``,
so that ``W ≈ weight @ linear_V.weight``. ``rank == "full"`` means no factorisation.
The compression code (``compression/lrd.py``) fills the factors; this mixin only
allocates them with the right shapes.
"""

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LowRankMixin:
    """Adds ``init_lrd(rank)`` / ``cancel_lrd()`` to a layer with a 2-D ``weight``.

    Subclasses define ``_lrd_rows`` (rows of the full weight) and ``_lrd_cols``
    (columns of the full weight, i.e. the input width of ``linear_V``).
    """

    @property
    def _lrd_rows(self) -> int:
        raise NotImplementedError

    @property
    def _lrd_cols(self) -> int:
        raise NotImplementedError

    def _new_weight(self, cols):
        weight = Parameter(
            torch.empty([self._lrd_rows, cols], device=self.weight.device, dtype=self.weight.dtype),
            requires_grad=True,
        )
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight

    def init_lrd(self, rank):
        """Allocate the factors for ``rank`` (an int), or keep the full weight (``None``/``"full"``)."""
        self.rank = "full" if rank is None else rank
        if not isinstance(rank, int):
            return
        # linear_V is a real nn.Linear so quantizers discover it like any other linear layer.
        if self.linear_V is None or self.linear_V.out_features != rank:
            self.linear_V = nn.Linear(self._lrd_cols, rank, bias=False,
                                      device=self.weight.device, dtype=self.weight.dtype)
        if self.weight.shape[1] != rank:
            self.weight = self._new_weight(rank)

    def cancel_lrd(self):
        """Drop the factors and re-allocate a full-size (uninitialised) weight."""
        self.rank = "full"
        self.linear_V = None
        if self.weight.shape[1] != self._lrd_cols:
            self.weight = self._new_weight(self._lrd_cols)


__all__ = ["LowRankMixin"]
