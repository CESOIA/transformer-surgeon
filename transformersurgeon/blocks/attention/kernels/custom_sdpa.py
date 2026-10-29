"""ExecuTorch's fused CPU attention kernel (``torch.ops.llama.custom_sdpa``).

The operators live in the ExecuTorch "llama" operator namespace; the name refers
to that library, not to a model family. They need float32, batched
``(1, seq, heads, head_dim)`` tensors and a cache written in place by
``torch.ops.llama.update_cache``, so this kernel keeps its own float32 caches
and bypasses the attention module's KV-cache atom.
"""

import torch

from .base import AttentionKernel


def _ensure_custom_sdpa_ops():
    """Register ``torch.ops.llama.custom_sdpa`` / ``update_cache`` (import side effect)."""
    try:
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "attn_impl='custom_sdpa' requires ExecuTorch's LLM custom-ops "
            "extension (torch.ops.llama.custom_sdpa / update_cache). Install "
            "the `executorch` extra so `executorch.extension.llm.custom_ops` "
            "is importable."
        ) from e


class CustomSdpaKernel(AttentionKernel):
    """Fused causal attention that only scores the filled cache positions.

    Told the real start position with ``is_causal=True``, the kernel scans
    positions ``[0, start_pos + q_len)`` instead of the whole cache, so decoding
    cost grows with the context rather than with ``max_cache_len``. The causal
    mask is derived by the kernel itself; the decoder's mask is not used.
    """

    name = "custom_sdpa"
    requires_cache = "mutable"
    owns_cache = True

    def _cache(self, attn, head_dim, device=None):
        return torch.zeros(1, attn.max_cache_length, attn.kv_num_heads, head_dim,
                           dtype=torch.float32, device=device)

    def setup(self, attn):
        _ensure_custom_sdpa_ops()
        attn.register_buffer("custom_sdpa_key_cache", self._cache(attn, attn.key_head_dim), persistent=False)
        attn.register_buffer("custom_sdpa_value_cache", self._cache(attn, attn.value_head_dim), persistent=False)

    def check_dims(self, attn):
        q_head_dim = attn.q_proj.out_features // attn.num_heads
        if q_head_dim != attn.value_head_dim:
            raise RuntimeError(
                "attn_impl='custom_sdpa' requires q_head_dim == value_head_dim "
                "(torch.ops.llama.custom_sdpa sizes its output like `query`, "
                f"not `value`); got q_head_dim={q_head_dim}, value_head_dim="
                f"{attn.value_head_dim}. This happens when RoPE-linked "
                "structured pruning shrank q/k's head_dim while v_proj stayed "
                "unpruned -- use attn_impl='manual' or 'sdpa' for such models."
            )

    def geometry_ok(self, attn):
        self.check_dims(attn)
        return (attn.custom_sdpa_key_cache.shape[-1] == attn.key_head_dim
                and attn.custom_sdpa_value_cache.shape[-1] == attn.value_head_dim)

    def finalize_geometry(self, attn):
        self.check_dims(attn)
        if attn.custom_sdpa_key_cache.shape[-1] != attn.key_head_dim:
            attn.custom_sdpa_key_cache = self._cache(attn, attn.key_head_dim, attn.custom_sdpa_key_cache.device)
        if attn.custom_sdpa_value_cache.shape[-1] != attn.value_head_dim:
            attn.custom_sdpa_value_cache = self._cache(attn, attn.value_head_dim,
                                                       attn.custom_sdpa_value_cache.device)

    def attend_with_own_cache(self, attn, q, k, v, pos_id):
        """Write ``k``/``v`` at ``pos_id`` and attend; returns ``(num_heads, q_len, value_head_dim)``."""
        orig_dtype = q.dtype
        start_pos = pos_id[0].item()
        q_b = q.unsqueeze(0).to(torch.float32)
        k_b = k.unsqueeze(0).to(torch.float32)
        v_b = v.unsqueeze(0).to(torch.float32)
        _ = torch.ops.llama.update_cache(k_b, attn.custom_sdpa_key_cache, start_pos)
        _ = torch.ops.llama.update_cache(v_b, attn.custom_sdpa_value_cache, start_pos)
        attn_output = torch.ops.llama.custom_sdpa(
            q_b, attn.custom_sdpa_key_cache, attn.custom_sdpa_value_cache,
            start_pos,          # real position: bounds the kernel's scan
            None,               # no explicit mask: derived from is_causal
            0.0,                # dropout
            True,               # is_causal
            attn.score_scale,
        )
        return attn_output.squeeze(0).to(orig_dtype).transpose(0, 1)


__all__ = ["CustomSdpaKernel"]
