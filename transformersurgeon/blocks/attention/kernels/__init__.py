"""Attention kernels: how scores and weighted sums are computed. One kernel per file."""
from .base import AttentionKernel
from .manual import ManualKernel, attention
from .sdpa import SdpaKernel
from .custom_sdpa import CustomSdpaKernel
from .paged import PagedKernel, paged_attention, KV_PAGE_SIZE, num_kv_pages, identity_page_table

ATTENTION_KERNELS = {
    kernel.name: kernel
    for kernel in (ManualKernel(), SdpaKernel(), CustomSdpaKernel(), PagedKernel())
}

__all__ = [
    "AttentionKernel", "ManualKernel", "SdpaKernel", "CustomSdpaKernel", "PagedKernel",
    "ATTENTION_KERNELS", "attention", "paged_attention", "KV_PAGE_SIZE", "num_kv_pages",
    "identity_page_table",
]
