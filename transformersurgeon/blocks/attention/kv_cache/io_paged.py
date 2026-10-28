"""Paged KV pool passed in and out of the graph (used with the ``paged`` kernel)."""

from ..kernels.paged import KV_PAGE_SIZE, num_kv_pages
from .base import KVCache


class PagedCache(KVCache):
    """One pool per layer, ``(2, num_pages, page_size, kv_heads, head_dim)`` (key pages,
    then value pages). Writing and reading are done by the ``paged`` kernel."""

    name = "io_paged"
    manifest_layout = "PAGED"
    multi_token = True
    requires_kernel = "paged"

    def shape(self, attn, which="kv"):
        return (2, num_kv_pages(attn.max_cache_length), KV_PAGE_SIZE, attn.kv_num_heads, attn.value_head_dim)


__all__ = ["PagedCache"]
