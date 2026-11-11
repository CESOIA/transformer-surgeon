"""Common interface of the KV-cache atoms."""

import torch


class KVCache:
    """How past keys and values are stored and updated during decoding.

    A KV-cache atom is a stateless strategy object owned by a causal attention
    module (which holds the actual buffers). Attributes:

    - ``graph_io``: the cache is an input and an output of the exported graph
      (``False`` means it is internal module state, mutated in place).
    - ``read_layout``: ``"shd"`` (sequence, heads, dim) or ``"hsd"`` (heads,
      sequence, dim) as seen by the attention kernel.
    - ``manifest_layout``: layout name recorded in export manifests.
    - ``multi_token``: whether one call may write several tokens (prefill).
    - ``requires_kernel``: the only attention kernel that works with this cache, or ``None``.
    - ``returns_rows``: the graph only reads the cache and returns the new stored rows, which
      the caller writes into its own cache (``False``: the graph returns the updated cache).
    """

    name = None
    graph_io = True
    read_layout = "shd"
    manifest_layout = "SHD"
    multi_token = False
    requires_kernel = None
    returns_rows = False

    def shape(self, attn, which):
        """Shape of the ``which`` (``"key"``/``"value"``, or their block scales ``"key_scale"``/
        ``"value_scale"``) cache of ``attn``. The last dim is the stored row width
        (``attn.cache_width``: the head dim, or its packed form)."""
        return (attn.max_cache_length, attn.kv_num_heads, attn.cache_width(which))

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        """Return the caches updated with the stored rows ``k``/``v`` (``(q_len, kv_heads, width)``)
        at ``write_index``."""
        raise NotImplementedError

    def read(self, cache):
        """The cache as seen by the kernel (``read_layout``)."""
        return cache

    def update(self, attn, k, v, write_index, key_cache, value_cache, dtype):
        """Store the new float rows ``k``/``v`` (``(q_len, kv_heads, head_dim)``) and return
        ``(key, value, key_out, value_out)``: the keys/values the kernel attends to, decoded to
        ``dtype`` in ``read_layout``, and the caches the module returns.

        Default: encode the new rows, write them, then decode the whole cache.
        """
        codec = attn.kv_codec
        key_cache, value_cache = self.write(attn, codec.encode(k), codec.encode(v), write_index, key_cache, value_cache)
        key = codec.decode(self.read(key_cache), dtype)
        value = codec.decode(self.read(value_cache), dtype)
        return key, value, key_cache, value_cache

    def decoder_mask(self, decoder, pos_id, in_seq_len):
        """Additive causal mask and the positions of the new tokens.

        Default: one new token at ``pos_id``; mask row ``pos_id`` of the
        precomputed ``(max_cache_len, max_cache_len)`` causal table, shape
        ``(1, max_cache_len)``.
        """
        attn_mask = torch.where(
            decoder.q_pos < decoder.k_pos, decoder.mask_penalty, torch.zeros_like(decoder.mask_penalty)
        )[pos_id].unsqueeze(0)
        return attn_mask, pos_id


__all__ = ["KVCache"]
