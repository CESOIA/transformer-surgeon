import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class EmbeddingCompressed(nn.Module):
    """
    Embedding lookup table with support for low-rank decomposition.

    ``weight`` is stored in the same native layout as ``nn.Embedding``
    (``[num_embeddings, embedding_dim]``), so checkpoints saved by a plain
    ``nn.Embedding`` load here without any shape translation (HF's
    ``from_pretrained`` shape-checks parameters before ``load_state_dict``
    hooks run, so a transposed convention would break loading).

    LRD factors ``weight ≈ US_r[num_embeddings, rank] @ V_r[rank,
    embedding_dim]`` -- ``LRDer`` (``compression/lrd.py``) works on this
    module unmodified: it only ever calls ``module.init_lrd``/``weight``/
    ``linear_V`` generically, and low-rank SVD factorization of a 2-D matrix
    doesn't care which axis is semantically "vocab" vs "embedding_dim".

    Structured pruning (the residual/embedding_dim axis) is NOT the row axis
    here (unlike ``LinearCompressed``, where the residual axis is
    ``out_features``/dim 0) -- it's dim 1. ``StructuredPruner``
    (``compression/structured_pruning.py``) has small, explicit branches for
    this module type to account for that.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 rank=None,
                 device=None,
                 dtype=None,
                 ):
        super().__init__()

        if embedding_dim <= 0:
            self.skip = True  # If embedding_dim is 0, skip the layer (fully pruned)
            return

        self.skip = False
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Aliases mirroring LinearCompressed's naming, for generic code that
        # introspects them (cosmetic bookkeeping only -- the actual pruning
        # axis is handled explicitly in StructuredPruner, not derived from
        # these).
        self.in_features = num_embeddings
        self.out_features = embedding_dim
        self.bias = None  # Embeddings never have a bias; kept for API parity with LinearCompressed

        self.weight = Parameter(
            torch.empty([num_embeddings, embedding_dim], device=device, dtype=dtype)
        )
        torch.nn.init.normal_(self.weight)

        self.linear_V = None  # Placeholder for the V factor (rank->embedding_dim) in low-rank decomposition
        self.init_lrd(rank)

    def init_lrd(self, rank):
        # Set rank and initialize linear_V for low-rank decomposition if needed
        self.rank = "full" if rank is None else rank
        if isinstance(rank, int):
            device = self.weight.device
            dtype = self.weight.dtype

            # V factor (rank -> embedding_dim), as a proper nn.Linear so that
            # torchao's quantize_() can discover and quantize it alongside
            # self.weight.
            if self.linear_V is None or self.linear_V.out_features != rank:
                self.linear_V = nn.Linear(self.embedding_dim, rank, bias=False,
                                          device=device, dtype=dtype)

            # Shrink weight to the small [num_embeddings, rank] table.
            if self.weight.shape[1] != self.rank:
                self.weight = Parameter(
                    torch.empty([self.num_embeddings, self.rank], device=device, dtype=dtype),
                    requires_grad=True)
                torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def cancel_lrd(self):
        self.rank = "full"
        device = self.weight.device
        dtype = self.weight.dtype

        if self.linear_V is not None:
            self.linear_V = None

        if self.weight.shape[1] != self.embedding_dim:
            self.weight = None
            self.weight = Parameter(
                torch.empty([self.num_embeddings, self.embedding_dim], device=device, dtype=dtype),
                requires_grad=True)
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.skip:
            return input

        if isinstance(self.rank, int):
            v = F.embedding(input, self.weight)  # [*, rank]  (lookup in the small table)
            return v @ self.linear_V.weight  # [*, rank] @ [rank, embedding_dim] = [*, embedding_dim]

        return F.embedding(input, self.weight)  # [*, embedding_dim]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"EmbeddingCompressed(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, rank={self.rank})"

__all__ = ["EmbeddingCompressed"]
