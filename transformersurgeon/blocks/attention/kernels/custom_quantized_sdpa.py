"""ExecuTorch's fused CPU attention over an int8 KV cache (``torch.ops.llama.custom_quantized_sdpa``)."""

import torch

from .custom_sdpa import CustomSdpaKernel


def _ensure_quantized_decomposed_ops():
    """Register ``torch.ops.quantized_decomposed`` (per-token int8 qparams / quantize) and, when ExecuTorch is
    installed, their out variants (``executorch.kernels.quantized``), which ``to_executorch()`` needs."""
    import torch.ao.quantization.fx._decomposed  # noqa: F401
    try:
        from executorch.kernels import quantized  # noqa: F401
    except ImportError:
        pass


def _quantize_per_token_int8(x):
    """Asymmetric int8 over the last dim, one (scale, zero point) per leading index: for a
    ``(1, seq, heads, head_dim)`` tensor that is per token and head. Returns (int8, float32 scale, int8 zero point)."""
    scale, zp = torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(x, torch.int8)
    q = torch.ops.quantized_decomposed.quantize_per_token(x, scale, zp, -128, 127, torch.int8)
    return q, scale.to(torch.float32), zp.to(torch.int8)


class CustomQuantizedSdpaKernel(CustomSdpaKernel):
    """:class:`CustomSdpaKernel` with an int8 cache: keys and values hold int8 codes with one float32 scale and
    int8 zero point per (token, kv head), appended with ``llama.update_cache``; attention reads them through
    ``llama.custom_quantized_sdpa`` (the query is quantised per token the same way). This is ExecuTorch's
    QuantizedKVCache + QuantizedSDPA: 1 byte per cached value instead of 4.

    The kernel keeps its own cache (``owns_cache``), so it is selected with ``attn_impl``, not with the
    ``kv_precision`` atoms (which encode rows for caches written through the KV-cache atom).
    """

    name = "custom_quantized_sdpa"

    def _cache(self, attn, head_dim, device=None):
        return torch.zeros(1, attn.max_cache_length, attn.kv_num_heads, head_dim, dtype=torch.int8, device=device)

    def setup(self, attn):
        super().setup(attn)
        _ensure_quantized_decomposed_ops()
        shape = (1, attn.max_cache_length, attn.kv_num_heads, 1)
        for which in ("key", "value"):
            attn.register_buffer(f"custom_sdpa_{which}_scales", torch.ones(shape, dtype=torch.float32),
                                 persistent=False)
            attn.register_buffer(f"custom_sdpa_{which}_zero_points", torch.zeros(shape, dtype=torch.int8),
                                 persistent=False)

    def attend_with_own_cache(self, attn, q, k, v, pos_id):
        """Quantise the new rows, append them with their scales and zero points, attend over the int8 cache;
        returns ``(num_heads, q_len, value_head_dim)``."""
        orig_dtype = q.dtype
        start_pos = pos_id[0].item()
        q_q, q_scale, q_zp = _quantize_per_token_int8(q.unsqueeze(0).to(torch.float32))
        for which, x in (("key", k), ("value", v)):
            x_q, x_scale, x_zp = _quantize_per_token_int8(x.unsqueeze(0).to(torch.float32))
            _ = torch.ops.llama.update_cache(x_q, getattr(attn, f"custom_sdpa_{which}_cache"), start_pos)
            _ = torch.ops.llama.update_cache(x_scale, getattr(attn, f"custom_sdpa_{which}_scales"), start_pos)
            _ = torch.ops.llama.update_cache(x_zp, getattr(attn, f"custom_sdpa_{which}_zero_points"), start_pos)
        attn_output = torch.ops.llama.custom_quantized_sdpa(
            q_q, attn.custom_sdpa_key_cache, attn.custom_sdpa_value_cache,
            start_pos, None, 0.0, True, attn.score_scale,
            q_zp, q_scale,
            attn.custom_sdpa_key_zero_points, attn.custom_sdpa_key_scales,
            attn.custom_sdpa_value_zero_points, attn.custom_sdpa_value_scales,
        )  # (1, q_len, num_heads, value_head_dim), float32
        return attn_output.squeeze(0).to(orig_dtype).transpose(0, 1)


__all__ = ["CustomQuantizedSdpaKernel"]
