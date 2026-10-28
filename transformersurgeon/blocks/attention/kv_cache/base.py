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
    """

    name = None
    graph_io = True
    read_layout = "shd"
    manifest_layout = "SHD"
    multi_token = False

    def shape(self, attn, which):
        """Shape of the ``which`` (``"key"``/``"value"``) cache of ``attn``."""
        head_dim = attn.key_head_dim if which == "key" else attn.value_head_dim
        return (attn.max_cache_length, attn.kv_num_heads, head_dim)

    def write(self, attn, k, v, write_index, key_cache, value_cache):
        """Return the caches updated with ``k``/``v`` (``(q_len, kv_heads, dim)``) at ``write_index``."""
        raise NotImplementedError

    def read(self, cache):
        """The cache as seen by the kernel (``read_layout``)."""
        return cache

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
