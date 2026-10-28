"""KV cache kept as internal module state and written in place."""

import torch

from .base import KVCache


class MutableCache(KVCache):
    """In-place ``index_copy_`` on the attention module's own buffers.

    The cache is not part of the graph inputs/outputs: ExecuTorch backends keep
    it as mutable state between calls.
    """

    name = "mutable"
    graph_io = False

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        with torch.no_grad():
            attn.key_cache.index_copy_(0, write_index, k)
            attn.value_cache.index_copy_(0, write_index, v)
        return attn.key_cache, attn.value_cache


__all__ = ["MutableCache"]
