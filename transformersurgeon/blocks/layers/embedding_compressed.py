"""Embedding table that can hold a low-rank factorisation."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .low_rank import LowRankMixin


class EmbeddingCompressed(LowRankMixin, nn.Module):
    """Embedding lookup table with optional low-rank factorisation.

    ``weight`` keeps the ``nn.Embedding`` layout ``[num_embeddings, embedding_dim]``
    so checkpoints of a plain ``nn.Embedding`` load unchanged. With a rank, the
    table becomes ``weight[num_embeddings, rank] @ linear_V.weight[rank, embedding_dim]``.

    Unlike :class:`LinearCompressed`, the axis shared with the rest of the network
    (the hidden size) is dimension 1 of ``weight``; structured pruning accounts for
    that. ``in_features``/``out_features``/``bias`` exist for interface parity with
    :class:`LinearCompressed`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, rank=None, device=None, dtype=None):
        super().__init__()
        if embedding_dim <= 0:
            self.skip = True
            self.num_embeddings, self.embedding_dim, self.rank = num_embeddings, 0, "full"
            return
        self.skip = False
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.in_features = num_embeddings
        self.out_features = embedding_dim
        self.bias = None
        self.weight = Parameter(torch.empty([num_embeddings, embedding_dim], device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight)
        self.linear_V = None
        self.init_lrd(rank)

    @property
    def _lrd_rows(self):
        return self.num_embeddings

    @property
    def _lrd_cols(self):
        return self.embedding_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.skip:
            return input
        if isinstance(self.rank, int):
            return F.embedding(input, self.weight) @ self.linear_V.weight
        return F.embedding(input, self.weight)

    def __repr__(self):
        return (f"EmbeddingCompressed(num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim}, rank={self.rank})")

    __str__ = __repr__


__all__ = ["EmbeddingCompressed"]
