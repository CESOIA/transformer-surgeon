"""KV-cache atoms: how past keys/values are stored and written. One atom per file."""
from .base import KVCache
from .mutable import MutableCache
from .io_scatter import ScatterCache
from .io_concat import ConcatCache
from .io_inplace import InplaceCache, kv_cache_update
from .io_paged import PagedCache

KV_CACHES = {
    cache.name: cache
    for cache in (MutableCache(), ScatterCache(), ConcatCache(), InplaceCache(), PagedCache())
}

__all__ = [
    "KVCache", "MutableCache", "ScatterCache", "ConcatCache", "InplaceCache", "PagedCache",
    "KV_CACHES", "kv_cache_update",
]
