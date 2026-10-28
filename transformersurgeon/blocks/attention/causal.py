"""Causal multi-head attention over a fixed-size KV cache (decoder attention)."""

import torch

from .base import MHABase
from .kv_cache import KV_CACHES


class MHACausal(MHABase):
    """Causal multi-head attention for decoders, composed of two selectable atoms:

    - a KV-cache atom (``cache_impl``, see ``kv_cache/``): how new keys/values are
      stored (internal state, scatter, one-hot select, in-place operator, pages);
    - an attention kernel (``attn_impl``, see ``kernels/``): how attention is
      computed (explicit, PyTorch SDPA, ExecuTorch fused kernel, paged kernel).

    Extra keyword arguments: ``cache_impl`` (default ``"mutable"``),
    ``max_cache_len`` (default 2048), ``add_batch_dim`` (accept and return a
    leading size-1 batch dimension), ``masked_softmax`` (replace masked scores
    instead of adding a large negative penalty; ``manual`` kernel only).

    ``forward`` returns ``output`` when the cache is internal state, otherwise
    ``(output, key_cache, value_cache)`` (``(output, kv_pool, None)`` for pages).
    """

    def __init__(self, embed_dim, num_heads, kv_num_heads=None, attn_impl=None, use_sdpa=False,
                 linear_specs=None, dtype=None, cache_impl="mutable", max_cache_len=2048,
                 add_batch_dim=False, masked_softmax=False):
        if cache_impl not in KV_CACHES:
            raise ValueError(f"Unsupported cache_impl {cache_impl!r}; expected one of {tuple(KV_CACHES)}")
        self.cache_impl = cache_impl
        super().__init__(embed_dim, num_heads, kv_num_heads=kv_num_heads, attn_impl=attn_impl,
                         use_sdpa=use_sdpa, linear_specs=linear_specs, dtype=dtype)
        self.kv_cache = KV_CACHES[cache_impl]
        self.add_batch_dim = add_batch_dim
        self.max_cache_length = max_cache_len
        self.masked_softmax = bool(masked_softmax)

        if self.kv_cache.requires_kernel not in (None, self.attn_impl) or (
                self.kernel.requires_cache == "io_paged" and cache_impl != "io_paged"):
            raise ValueError("attn_impl='paged' and cache_impl='io_paged' go together.")
        if self.masked_softmax and not self.kernel.supports_masked_softmax:
            raise ValueError("masked_softmax=True needs attn_impl='manual'.")
        if self.kernel.requires_cache not in (None, cache_impl):
            raise ValueError(
                f"attn_impl={self.attn_impl!r} requires cache_impl={self.kernel.requires_cache!r}"
                + (": torch.ops.llama.update_cache in-place-mutates a dedicated batched fp32 cache buffer, "
                   "which has no functional (io_scatter/io_concat) equivalent." if self.kernel.owns_cache else ".")
            )

        # Internal buffers back the "mutable" cache and are the initial cache of the
        # io_* caches. They are resized once by finalize_cache_geometry() after pruned
        # weights are loaded. The paged pool lives outside the module.
        empty = (0,) if cache_impl == "io_paged" else None
        self.register_buffer("key_cache", torch.zeros(empty or self.cache_shape("key"), dtype=dtype),
                             persistent=False)
        self.register_buffer("value_cache", torch.zeros(empty or self.cache_shape("value"), dtype=dtype),
                             persistent=False)
        self.kernel.setup(self)

    def _default_attn_impl(self):
        return "paged" if self.cache_impl == "io_paged" else super()._default_attn_impl()

    def cache_shape(self, which):
        """Shape of the ``which`` (``"key"``, ``"value"``; ``"kv"`` for pages) cache."""
        return self.kv_cache.shape(self, which)

    def _cache_geometry_ok(self):
        if self.cache_impl == "io_paged":
            return True
        ok = (tuple(self.key_cache.shape) == self.cache_shape("key")
              and tuple(self.value_cache.shape) == self.cache_shape("value"))
        return self.kernel.geometry_ok(self) and ok

    def _check_cache_geometry(self):
        """The internal buffers must already match the (possibly pruned) head dims:
        resizing them inside ``forward`` would export as a fresh zero cache per step."""
        if not self._cache_geometry_ok():
            raise RuntimeError(
                "KV cache buffers do not match the pruned key/value head dims "
                f"(key_cache {tuple(self.key_cache.shape)} vs {self.cache_shape('key')}, "
                f"value_cache {tuple(self.value_cache.shape)} vs {self.cache_shape('value')}). "
                "Call finalize_cache_geometry() once after loading pruned weights "
                "(convert_for_export does this)."
            )

    def finalize_cache_geometry(self):
        """Resize the internal KV buffers to the current (possibly pruned) head dims.
        Conversion-time hook: called once after weights are loaded."""
        if self.cache_impl == "io_paged":
            return
        for which in ("key", "value"):
            buffer = getattr(self, f"{which}_cache")
            if tuple(buffer.shape) != self.cache_shape(which):
                setattr(self, f"{which}_cache",
                        torch.zeros(self.cache_shape(which), dtype=buffer.dtype, device=buffer.device))
        self.kernel.finalize_geometry(self)

    def forward(self, x, pos_id, attn_mask, key_cache=None, value_cache=None, rope=None, paged=None):
        """
        Args:
            x: ``(q_len, embed_dim)``, or ``(1, q_len, embed_dim)`` with ``add_batch_dim``.
            pos_id: ``(1,)`` position of the first new token.
            attn_mask: additive causal mask built once per step by the decoder.
            key_cache/value_cache: incoming caches for graph-I/O caches; the internal
                buffers are used when omitted. For pages, ``key_cache`` is the pool.
            rope: ``(cos, sin)`` already indexed at the new positions.
            paged: per-step inputs of the paged kernel (see ``kernels/paged.py``).
        """
        in_seq_len = x.size(1) if self.add_batch_dim else x.size(0)

        if self.kernel.fuses_projection:
            return self.kernel.forward_fused(self, x, in_seq_len, key_cache, paged)

        self._check_cache_geometry()
        if key_cache is None:
            key_cache = self.key_cache
        if value_cache is None:
            value_cache = self.value_cache

        # With add_batch_dim the leading size-1 dim is absorbed by the views below.
        q, k, v = self._project_qkv(x, in_seq_len)
        q, k = self._apply_rope(q, k, rope)

        if self.kernel.owns_cache:
            attn_output = self.kernel.attend_with_own_cache(self, q, k, v, pos_id)
        else:
            write_index = torch.clamp(pos_id, 0, self.max_cache_length - 1).long()
            key_cache, value_cache = self.kv_cache.write(self, k, v, write_index, key_cache, value_cache)
            attn_output = self.kernel.attend(self, q, self.kv_cache.read(key_cache), self.kv_cache.read(value_cache),
                                             attn_mask, kv_layout=self.kv_cache.read_layout)

        # Width is num_heads * value_head_dim (out_proj's input follows value pruning).
        width = self.num_heads * self.value_head_dim
        out_shape = (1, in_seq_len, width) if self.add_batch_dim else (in_seq_len, width)
        output = self.out_proj(attn_output.transpose(0, 1).reshape(out_shape))

        if not self.kv_cache.graph_io:
            return output
        return output, key_cache, value_cache


CACHE_IMPLS = tuple(KV_CACHES)

__all__ = ["MHACausal", "CACHE_IMPLS"]
