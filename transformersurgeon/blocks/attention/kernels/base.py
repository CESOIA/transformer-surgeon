"""Common interface of the attention kernels."""


class AttentionKernel:
    """How the attention scores and the weighted sum are computed.

    A kernel is a stateless strategy object owned by an attention module. Class
    attributes describe what it needs; the attention module checks them:

    - ``requires_cache``: name of the only KV-cache atom it works with, or ``None``.
    - ``owns_cache``: the kernel keeps and writes its own cache (the KV-cache atom
      is then bypassed).
    - ``fuses_projection``: the kernel consumes the packed q|k|v projections and
      applies the rotary embedding itself.
    - ``supports_masked_softmax``: the kernel implements ``masked_softmax``.
    """

    name = None
    requires_cache = None
    owns_cache = False
    fuses_projection = False
    supports_masked_softmax = False

    def setup(self, attn):
        """Register any buffer the kernel needs on the attention module."""

    def geometry_ok(self, attn):
        """Whether the kernel's own buffers match the (possibly pruned) head dims."""
        return True

    def finalize_geometry(self, attn):
        """Resize the kernel's own buffers after pruned weights are loaded."""

    def attend(self, attn, q, key, value, attn_mask=None, kv_layout="shd"):
        """Return ``(num_heads, q_len, value_head_dim)``.

        ``q`` is ``(q_len, num_heads, head_dim)``; ``key``/``value`` are
        ``(kv_len, kv_heads, head_dim)`` (``kv_layout="shd"``) or
        ``(kv_heads, kv_len, head_dim)`` (``kv_layout="hsd"``).
        """
        raise NotImplementedError


__all__ = ["AttentionKernel"]
