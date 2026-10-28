"""KV-cache update op for the ``io_inplace`` cache implementation.

``torch.ops.tsurgeon.kv_cache_update(cache, update, write_index)`` returns
``cache`` with ``update`` written along the sequence axis (dim 2) starting at
``write_index``. It is functional in eager PyTorch (the input is not mutated),
which keeps ``torch.export`` graphs pure, while its ONNX translation
(``export/onnx``) is the opset-24 ``TensorScatter`` op: TensorRT maps that to
its in-place KV-cache update layer and aliases the output to the input buffer,
so no per-step cache copy is made.

Layout is BHSD: cache ``(1, kv_heads, max_cache_len, head_dim)``, update
``(1, kv_heads, seq_len, head_dim)``, write_index ``(1,)`` int64.
"""

import torch

from .base import KVCache


@torch.library.custom_op("tsurgeon::kv_cache_update", mutates_args=())
def kv_cache_update(cache: torch.Tensor, update: torch.Tensor, write_index: torch.Tensor) -> torch.Tensor:
    positions = write_index + torch.arange(update.shape[2], device=cache.device)
    return cache.index_copy(2, positions, update)


@kv_cache_update.register_fake
def _(cache, update, write_index):
    return torch.empty_like(cache)


class InplaceCache(KVCache):
    """Cache as graph input/output in ``(1, kv_heads, max_cache_len, head_dim)`` layout,
    written by :func:`kv_cache_update`. Accepts several new tokens per call (prefill)."""

    name = "io_inplace"
    read_layout = "hsd"
    manifest_layout = "BHSD"
    multi_token = True

    def shape(self, attn, which):
        head_dim = attn.key_head_dim if which == "key" else attn.value_head_dim
        return (1, attn.kv_num_heads, attn.max_cache_length, head_dim)

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        key_cache = kv_cache_update(key_cache, k.transpose(0, 1).unsqueeze(0), write_index)
        value_cache = kv_cache_update(value_cache, v.transpose(0, 1).unsqueeze(0), write_index)
        return key_cache, value_cache

    def read(self, cache):
        return cache[0]

    def decoder_mask(self, decoder, pos_id, in_seq_len):
        positions = pos_id + torch.arange(in_seq_len, device=pos_id.device)
        attn_mask = torch.where(
            decoder.k_pos > positions[:, None], decoder.mask_penalty, torch.zeros_like(decoder.mask_penalty)
        ).unsqueeze(0)
        return attn_mask, positions


__all__ = ["kv_cache_update", "InplaceCache"]
