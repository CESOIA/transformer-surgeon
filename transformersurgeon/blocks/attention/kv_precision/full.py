"""Keys and values stored in the model dtype (no quantisation)."""

from .base import KVPrecision


class FullPrecision(KVPrecision):
    """The cache keeps the model dtype; encoding and decoding are the identity
    (no operation is added to the graph)."""

    name = "full"

    def storage_dtype(self, dtype):
        return dtype

    def row_width(self, head_dim):
        return head_dim

    def encode(self, x):
        return x

    def decode(self, rows, dtype):
        return rows


__all__ = ["FullPrecision"]
