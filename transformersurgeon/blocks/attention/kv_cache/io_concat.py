"""KV cache passed in and out of the graph, updated without any scatter operation."""

import torch

from .base import KVCache


class ConcatCache(KVCache):
    """One-hot position mask plus ``where``: a fallback for backends without a scatter
    operation. Writes exactly one token per call."""

    name = "io_concat"

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        idx = torch.arange(attn.max_cache_length, device=k.device)
        sel = (idx == write_index).view(attn.max_cache_length, 1, 1)
        k_row = k.reshape(1, attn.kv_num_heads, attn.key_head_dim)
        v_row = v.reshape(1, attn.kv_num_heads, attn.value_head_dim)
        return torch.where(sel, k_row, key_cache), torch.where(sel, v_row, value_cache)


__all__ = ["ConcatCache"]
