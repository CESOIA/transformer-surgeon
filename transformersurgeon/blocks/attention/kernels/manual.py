"""Attention computed with explicit matrix products and softmax."""

import math

import torch

from .base import AttentionKernel


def attention(query, key, value, attn_mask=None, kv_layout="shd", masked_softmax=False, scale=None):
    """
    Explicit implementation of scaled dot-product attention.
    This is needed for cases where torch's built-in SDPA is not suited for model export (e.g., ONNX).
    It supports GQA automatically by handling different head dimensions for query and key/value.
    key_cache and value_cache are provided separately to minimize the concatenation overhead. They are used only if is_causal is False.

    Args:
        query: Tensor of shape (q_seq_length, q_head_num, q_head_dim)
        key:   Tensor of shape (kv_seq_length, kv_head_num, k_head_dim), or
               (kv_head_num, kv_seq_length, k_head_dim) with kv_layout="hsd"
        value: Tensor of shape (kv_seq_length, kv_head_num, v_head_dim), or
               (kv_head_num, kv_seq_length, v_head_dim) with kv_layout="hsd"
        attn_mask: Optional additive mask broadcastable against the scores'
            trailing (q_seq_length, kv_seq_length) dims -- e.g. (1, kv_seq_length)
            as built by TransformerDecoder, or (1, 1, q_seq_length, kv_seq_length).
        kv_layout: "shd" (sequence-major, the unbatched caches) or "hsd"
            (head-major, the io_inplace BHSD cache with its batch dim dropped).
        masked_softmax: instead of adding the (large negative) mask to the scores,
            replace masked scores with (row minimum - 20) -- the vendor
            enable_masked_softmax form. Mathematically the same attention (masked
            weights ~e^-20 of the smallest live one, vs e^-65504), but the scores
            keep their own range: with an additive -65504 a 16-bit quantized score
            tensor spans ~[-65504, 1e4], i.e. a ~1.15-logit quantization step.
            Both reduce to the same contraction below; the two transposes
            compose into one permute that exporters fold into the matmul.

    Returns:
        Tensor of shape (q_head_num, q_seq_length, v_head_dim).
    """
    if kv_layout == "hsd":
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
    q_seq_len, q_head_num, q_head_dim = query.size()
    _, kv_head_num, k_head_dim = key.size()
    _, _, v_head_dim = value.size()
    group_size = q_head_num // kv_head_num

    dtype = query.dtype
    # Score scale: callers pass the *original* head dim's 1/sqrt(d) (MHABase.score_scale),
    # so pruning q/k channels removes their terms from q.k without changing the softmax
    # temperature (hard pruning == soft pruning == the pruned HF model). The q_head_dim
    # fallback only serves callers that pass nothing. Key and value keep their own last
    # dims (value may differ under v pruning).
    if scale is None:
        scale = 1.0 / math.sqrt(q_head_dim)

    # GQA by broadcast, not by materialization: group the query heads under their
    # kv head and let matmul broadcast the single key/value head across the group,
    # so the KV cache is never copied into a repeat_interleave'd (q_head_num-wide)
    # tensor.
    #
    # This ordering matters a great deal on QNN/HTP. The previous formulation
    # expanded first and transposed second, which meant the Transpose ran over a
    # group_size-times-larger tensor: for a 2048-slot cache with 14 q heads / 2 kv
    # heads that is a 7.3 MB intermediate, too big for VTCM, so the HTP compiler
    # spilled it to DDR. Measured per attention op at that size (AOT "DDR
    # bandwidth summary", SM8650): 9.18 MB spill + 9.18 MB fill + 9.2 MB
    # read/write, versus 0 MB spill and ~0.02 MB read here -- and the key permute
    # is fused into the matmul's input layout rather than materialized at all.
    # The matching vendor reference is
    # executorch/examples/qualcomm/oss_scripts/llama/model/static_llama.py, which
    # likewise keeps the cache in matmul layout and never permutes a cache-sized
    # tensor.
    #
    # The math is unchanged -- this is the same contraction, so results match the
    # old path to float round-off (~1e-9 in fp32) -- and XNNPACK partitions it at
    # least as well (one fewer expand_copy outside the delegate).
    query = query.transpose(0, 1).reshape(kv_head_num, group_size, q_seq_len, q_head_dim)
    key   = key.permute(1, 2, 0).unsqueeze(1)   # (kv_head_num, 1, k_head_dim, kv_len)
    value = value.transpose(0, 1).unsqueeze(1)  # (kv_head_num, 1, kv_len, v_head_dim)

    scores = torch.matmul(query * scale, key)   # (kv_head_num, group, q_seq_len, kv_len)

    if attn_mask is not None:
        if masked_softmax:
            floor = torch.amin(scores, dim=-1, keepdim=True) - 20
            scores = torch.where(attn_mask == 0, scores, floor)
        else:
            scores = scores + attn_mask

    scores = torch.nn.functional.softmax(scores, dim=-1)

    attn_output = torch.matmul(scores.to(dtype), value)  # (kv_head_num, group, q_seq_len, v_head_dim)
    attn_output = attn_output.reshape(q_head_num, q_seq_len, v_head_dim)

    return attn_output.to(dtype)  # (q_head_num, q_seq_len, v_head_dim)


class ManualKernel(AttentionKernel):
    """Explicit attention (:func:`attention`). Works on every backend and is the only
    kernel that implements ``masked_softmax``."""

    name = "manual"
    supports_masked_softmax = True

    def attend(self, attn, q, key, value, attn_mask=None, kv_layout="shd"):
        return attention(q, key, value, attn_mask, kv_layout=kv_layout,
                         masked_softmax=getattr(attn, "masked_softmax", False), scale=attn.score_scale)


__all__ = ["attention", "ManualKernel"]
