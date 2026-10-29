"""Attention through ``torch.nn.functional.scaled_dot_product_attention``."""

import torch.nn.functional as F

from .base import AttentionKernel


class SdpaKernel(AttentionKernel):
    """PyTorch's fused scaled-dot-product attention (grouped-query aware)."""

    name = "sdpa"

    def attend(self, attn, q, key, value, attn_mask=None, kv_layout="shd"):
        q_t = q.transpose(0, 1)
        k_t = key if kv_layout == "hsd" else key.transpose(0, 1)
        v_t = value if kv_layout == "hsd" else value.transpose(0, 1)
        return F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, is_causal=False,
                                              enable_gqa=True, scale=attn.score_scale)


__all__ = ["SdpaKernel"]
