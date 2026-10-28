"""Bidirectional multi-head attention without cache (encoder attention)."""

from .base import MHABase


class MHAEncoder(MHABase):
    """Every token attends to every token; no cache, no causal mask.

    Supports the kernels that need neither a cache nor fused projections
    (``manual``, ``sdpa``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kernel.requires_cache or self.kernel.owns_cache or self.kernel.fuses_projection:
            raise ValueError(f"attn_impl={self.attn_impl!r} is not available for encoder attention.")

    def forward(self, x, rope=None):
        seq_length = x.size(0)
        q, k, v = self._project_qkv(x, seq_length)
        q, k = self._apply_rope(q, k, rope)
        attn_output = self.kernel.attend(self, q, k, v)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.num_heads * self.value_head_dim)
        return self.out_proj(attn_output)


__all__ = ["MHAEncoder"]
