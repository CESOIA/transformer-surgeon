"""Keys and values stored as 8-bit codes with one scale per row."""

from .integer import SCALE_BYTES, IntegerKVPrecision


class Int8Precision(IntegerKVPrecision):
    """One byte per element (codes in [-127, 127], offset by 128) plus the two scale bytes."""

    name = "int8"
    bits = 8

    def row_width(self, head_dim):
        return head_dim + SCALE_BYTES

    def _pack(self, codes):
        return codes

    def _unpack(self, packed):
        return packed


__all__ = ["Int8Precision"]
