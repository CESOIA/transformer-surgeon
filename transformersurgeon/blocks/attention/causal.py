"""Causal multi-head attention over a fixed-size KV cache (decoder attention)."""

import torch

from ...utils.devices import device_zeros
from .base import MHABase
from .kv_cache import KV_CACHES
from .kv_precision import KV_PRECISIONS


class MHACausal(MHABase):
    """Causal multi-head attention for decoders, composed of three selectable atoms:

    - a KV-cache atom (``cache_impl``, see ``kv_cache/``): how new keys/values are
      stored (internal state, scatter, one-hot select, in-place operator, pages);
    - a KV-precision atom (``kv_precision``, see ``kv_precision/``): what one
      stored row holds (the model dtype, int8/int4 codes with one scale per
      row, FP8 numbers, or NVFP4 numbers with a second tensor of block scales);
      the kernel always reads decoded float rows;
    - an attention kernel (``attn_impl``, see ``kernels/``): how attention is
      computed (explicit, PyTorch SDPA, ExecuTorch fused kernel, paged kernel).

    Extra keyword arguments: ``cache_impl`` (default ``"mutable"``),
    ``max_cache_len`` (default 2048), ``add_batch_dim`` (accept and return a
    leading size-1 batch dimension), ``masked_softmax`` (replace masked scores
    instead of adding a large negative penalty; ``manual`` kernel only),
    ``kv_precision`` (default ``"full"``; the others need a kernel that reads the
    cache through the KV-cache atom, not ``custom_sdpa``/``paged``, and ``"fp8"``
    and ``"nvfp4"`` need ``cache_impl="io_rows"``).

    ``forward`` returns ``output`` when the cache is internal state, otherwise
    ``(output, key_cache, value_cache)`` (``(output, kv_pool, None)`` for pages; the
    new rows only for ``io_rows``). With ``kv_precision="nvfp4"`` each key/value
    cache, in and out, is the pair ``(rows, scales)``.
    """

    # Role read by exporters that treat whole atoms specially (e.g. the QNN
    # exporter keeps the attention core in fp16 under quant_scope="all").
    EXPORT_ROLE = "attention"

    def __init__(self, embed_dim, num_heads, kv_num_heads=None, attn_impl=None, use_sdpa=False,
                 linear_specs=None, dtype=None, cache_impl="mutable", max_cache_len=2048,
                 add_batch_dim=False, masked_softmax=False, kv_precision="full"):
        if cache_impl not in KV_CACHES:
            raise ValueError(f"Unsupported cache_impl {cache_impl!r}; expected one of {tuple(KV_CACHES)}")
        if kv_precision not in KV_PRECISIONS:
            raise ValueError(f"Unsupported kv_precision {kv_precision!r}; expected one of {tuple(KV_PRECISIONS)}")
        self.cache_impl = cache_impl
        super().__init__(embed_dim, num_heads, kv_num_heads=kv_num_heads, attn_impl=attn_impl,
                         use_sdpa=use_sdpa, linear_specs=linear_specs, dtype=dtype)
        self.kv_cache = KV_CACHES[cache_impl]
        self.kv_precision = kv_precision
        self.kv_codec = KV_PRECISIONS[kv_precision]
        self.add_batch_dim = add_batch_dim
        self.max_cache_length = max_cache_len
        self.masked_softmax = bool(masked_softmax)

        if kv_precision != "full" and (self.kernel.owns_cache or self.kernel.fuses_projection):
            raise ValueError(f"kv_precision={kv_precision!r} needs a cache written and read through the KV-cache "
                             f"atom; attn_impl={self.attn_impl!r} keeps its own float cache.")
        if self.kv_codec.requires_cache not in (None, cache_impl):
            raise ValueError(f"kv_precision={kv_precision!r} needs cache_impl={self.kv_codec.requires_cache!r}: "
                             "its rows cannot be moved by a scatter or select inside the graph.")

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
        storage = self.kv_codec.storage_dtype(dtype)
        self.register_buffer("key_cache", device_zeros(empty or self.cache_shape("key"), storage), persistent=False)
        self.register_buffer("value_cache", device_zeros(empty or self.cache_shape("value"), storage),
                             persistent=False)
        if self.cache_scale_dtype is not None:
            for which in ("key", "value"):
                self.register_buffer(f"{which}_cache_scale", device_zeros(self.cache_shape(f"{which}_scale"),
                                                                          self.cache_scale_dtype),
                                     persistent=False)
        self.kernel.setup(self)

    def _default_attn_impl(self):
        return "paged" if self.cache_impl == "io_paged" else super()._default_attn_impl()

    def cache_shape(self, which):
        """Shape of the ``which`` (``"key"``, ``"value"``, their block scales ``"key_scale"``,
        ``"value_scale"``; ``"kv"`` for pages) cache."""
        return self.kv_cache.shape(self, which)

    def cache_width(self, which):
        """Stored width of one ``which`` row: for ``"key"``/``"value"`` the per-head dim, or its
        packed form for a quantised ``kv_precision``; for ``"key_scale"``/``"value_scale"`` the
        number of block scales."""
        head_dim = self.key_head_dim if which.startswith("key") else self.value_head_dim
        if which.endswith("_scale"):
            return self.kv_codec.scale_width(head_dim)
        return self.kv_codec.row_width(head_dim)

    @property
    def cache_dtype(self):
        """dtype of the stored key/value caches (``uint8`` for an integer ``kv_precision``,
        ``float8_e4m3fn`` for ``fp8``, ``float4_e2m1fn_x2`` for ``nvfp4``)."""
        return self.key_cache.dtype

    @property
    def cache_scale_dtype(self):
        """dtype of the block scales stored next to each cache (``float8_e4m3fn`` for ``nvfp4``),
        or ``None`` when the rows hold everything."""
        return self.kv_codec.scale_dtype

    def zero_cache(self, which):
        """An empty ``which`` (``"key"``/``"value"``) cache in the form ``forward`` takes it: one
        tensor, or ``(rows, scales)`` with block scales."""
        rows = device_zeros(self.cache_shape(which), self.cache_dtype)
        if self.cache_scale_dtype is None:
            return rows
        return rows, device_zeros(self.cache_shape(f"{which}_scale"), self.cache_scale_dtype)

    def _cache_buffers(self):
        """``(buffer name, cache kind)`` of the internal cache buffers."""
        buffers = [("key_cache", "key"), ("value_cache", "value")]
        if self.cache_scale_dtype is not None:
            buffers += [("key_cache_scale", "key_scale"), ("value_cache_scale", "value_scale")]
        return buffers

    def _internal_cache(self, which):
        """The internal ``which`` buffer(s) in the form ``forward`` takes."""
        if self.cache_scale_dtype is None:
            return getattr(self, f"{which}_cache")
        return getattr(self, f"{which}_cache"), getattr(self, f"{which}_cache_scale")

    def _cache_geometry_ok(self):
        if self.cache_impl == "io_paged":
            return True
        ok = all(tuple(getattr(self, name).shape) == self.cache_shape(kind) for name, kind in self._cache_buffers())
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
        for name, kind in self._cache_buffers():
            buffer = getattr(self, name)
            if tuple(buffer.shape) != self.cache_shape(kind):
                setattr(self, name, device_zeros(self.cache_shape(kind), buffer.dtype, buffer.device))
        self.kernel.finalize_geometry(self)

    def forward(self, x, pos_id, attn_mask, key_cache=None, value_cache=None, rope=None, paged=None):
        """
        Args:
            x: ``(q_len, embed_dim)``, or ``(1, q_len, embed_dim)`` with ``add_batch_dim``.
            pos_id: ``(1,)`` position of the first new token.
            attn_mask: additive causal mask built once per step by the decoder.
            key_cache: incoming key cache for graph-I/O caches (the internal buffer
                is used when omitted; ``(rows, scales)`` with block scales). For pages,
                the pool.
            value_cache: incoming value cache, as ``key_cache`` (unused for pages).
            rope: ``(cos, sin)`` already indexed at the new positions.
            paged: per-step inputs of the paged kernel (see ``kernels/paged.py``).
        """
        in_seq_len = x.size(1) if self.add_batch_dim else x.size(0)

        if self.kernel.fuses_projection:
            return self.kernel.forward_fused(self, x, in_seq_len, key_cache, paged)

        self._check_cache_geometry()
        if key_cache is None:
            key_cache = self._internal_cache("key")
        if value_cache is None:
            value_cache = self._internal_cache("value")

        # With add_batch_dim the leading size-1 dim is absorbed by the views below.
        q, k, v = self._project_qkv(x, in_seq_len)
        q, k = self._apply_rope(q, k, rope)

        if self.kernel.owns_cache:
            attn_output = self.kernel.attend_with_own_cache(self, q, k, v, pos_id)
        else:
            write_index = torch.clamp(pos_id, 0, self.max_cache_length - 1).long()
            key, value, key_cache, value_cache = self.kv_cache.update(self, k, v, write_index, key_cache,
                                                                      value_cache, q.dtype)
            attn_output = self.kernel.attend(self, q, key, value, attn_mask, kv_layout=self.kv_cache.read_layout)

        # Width is num_heads * value_head_dim (out_proj's input follows value pruning).
        width = self.num_heads * self.value_head_dim
        out_shape = (1, in_seq_len, width) if self.add_batch_dim else (in_seq_len, width)
        output = self.out_proj(attn_output.transpose(0, 1).reshape(out_shape))

        if not self.kv_cache.graph_io:
            return output
        return output, key_cache, value_cache


CACHE_IMPLS = tuple(KV_CACHES)

__all__ = ["MHACausal", "CACHE_IMPLS"]
