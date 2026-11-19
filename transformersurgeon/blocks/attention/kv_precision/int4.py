"""Keys and values stored as 4-bit codes, two per byte, with one scale per row."""

import torch

from .integer import SCALE_BYTES, IntegerKVPrecision


class Int4Precision(IntegerKVPrecision):
    """Codes in [-7, 7] offset by 8; elements ``2i`` and ``2i + 1`` share byte ``i``
    as ``low + 16 * high``, so a row takes ``head_dim / 2`` bytes plus the two
    scale bytes. Needs an even head dim."""

    name = "int4"
    bits = 4

    def row_width(self, head_dim):
        if head_dim % 2:
            raise ValueError(f"kv_precision='int4' packs two elements per byte and needs an even head dim, "
                             f"got {head_dim}.")
        return head_dim // 2 + SCALE_BYTES

    def _pack(self, codes):
        pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
        return pairs[..., 0] + 16.0 * pairs[..., 1]

    def _unpack(self, packed):
        high = torch.floor(packed / 16.0)
        low = packed - 16.0 * high
        return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], 2 * packed.shape[-1])


__all__ = ["Int4Precision"]
