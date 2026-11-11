"""KV cache read by the graph and written by the caller (the graph returns only the new rows)."""

import torch

from .io_inplace import InplaceCache


class RowsCache(InplaceCache):
    """Cache as a graph input in ``(1, kv_heads, max_cache_len, width)`` layout that the graph
    only reads. The graph outputs the new stored rows, ``(1, kv_heads, q_len, width)``, and the
    caller writes them into its cache at ``pos_id`` (``export.manifest.write_cache_rows``).

    The stored cache is decoded first and the new rows are placed in the decoded copy, so
    stored rows never pass through a scatter or select: this is how TensorRT holds caches it
    cannot update itself (FP8 and FP4 tensors exist only between a quantize and a dequantize
    layer, uint8 only at the graph boundary). Accepts several new tokens per call (prefill).
    With block scales (``kv_precision="nvfp4"``) each cache and each returned set of rows
    is a ``(rows, scales)`` pair.

    The decoded new rows are placed with a gather and a select on the slot index, not a
    scatter: TensorRT 11.4 cannot compile an FP4 dequantize feeding a scatter when one token
    is written (decode profile).
    """

    name = "io_rows"
    returns_rows = True

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        raise NotImplementedError("io_rows caches are written by the caller; see RowsCache.update().")

    def update(self, attn, k, v, write_index, key_cache, value_cache, dtype):
        codec = attn.kv_codec
        # Slot s takes new row s - write_index when that is in [0, q_len).
        offset = torch.arange(attn.max_cache_length, device=k.device) - write_index
        new_slot = ((offset >= 0) & (offset < k.shape[0])).view(1, 1, -1, 1)
        source = offset.clamp(0, k.shape[0] - 1)
        decoded, rows = [], []
        for new, cache in ((k, key_cache), (v, value_cache)):
            stored = codec.encode(new.transpose(0, 1).unsqueeze(0))     # (1, kv_heads, q_len, width)
            placed = codec.decode(stored, dtype).index_select(2, source)
            merged = torch.where(new_slot, placed, codec.decode(cache, dtype))
            decoded.append(merged[0])
            rows.append(stored)
        return decoded[0], decoded[1], rows[0], rows[1]


__all__ = ["RowsCache"]
