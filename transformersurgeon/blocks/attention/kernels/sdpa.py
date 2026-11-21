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
        # A leading batch of 1: the ONNX translation (opset 23 Attention) takes 4-D tensors only,
        # and refuses enable_gqa when query and key/value have the same number of heads.
        gqa = q_t.shape[0] != k_t.shape[0]
        out = F.scaled_dot_product_attention(q_t.unsqueeze(0), k_t.unsqueeze(0), v_t.unsqueeze(0),
                                             attn_mask=attn_mask, is_causal=False, enable_gqa=gqa,
                                             scale=attn.score_scale)
        return out.squeeze(0)


__all__ = ["SdpaKernel"]
