"""KV cache read by the graph and written by the caller (the graph returns only the new rows)."""

import torch

from .io_inplace import InplaceCache


class RowsCache(InplaceCache):
    """Cache as a graph input in ``(1, kv_heads, max_cache_len, width)`` layout that the graph
    only reads. The graph outputs the new stored rows, ``(1, kv_heads, q_len, width)``, and the
    caller writes them into its cache at ``pos_id`` (``export.manifest.write_cache_rows``).

    The stored cache is decoded first and the new rows are placed in the decoded copy, so
    stored rows never pass through a scatter or select: this is how TensorRT holds caches it
    cannot update itself (FP8 tensors exist only between a quantize and a dequantize layer,
    uint8 only at the graph boundary). Accepts several new tokens per call (prefill).
    """

    name = "io_rows"

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        raise NotImplementedError("io_rows caches are written by the caller; see RowsCache.update().")

    def update(self, attn, k, v, write_index, key_cache, value_cache, dtype):
        codec = attn.kv_codec
        positions = write_index + torch.arange(k.shape[0], device=k.device)
        decoded, rows = [], []
        for new, cache in ((k, key_cache), (v, value_cache)):
            stored = codec.encode(new.transpose(0, 1).unsqueeze(0))     # (1, kv_heads, q_len, width)
            merged = codec.decode(cache, dtype).index_copy(2, positions, codec.decode(stored, dtype))
            decoded.append(merged[0])
            rows.append(stored)
        return decoded[0], decoded[1], rows[0], rows[1]


__all__ = ["RowsCache"]
