"""KV cache passed in and out of the graph, updated with a functional scatter."""

from .base import KVCache


class ScatterCache(KVCache):
    """Functional ``index_put``: lowers to ScatterND-style operations."""

    name = "io_scatter"

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        return key_cache.index_put((write_index,), k), value_cache.index_put((write_index,), v)


__all__ = ["ScatterCache"]
