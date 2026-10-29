"""Fused paged-KV decode attention op for the ``io_paged`` cache implementation.

``torch.ops.tsurgeon.paged_attention`` takes the packed q/k/v projections of the
new tokens, applies RoPE to q and k, writes k and v into a paged KV pool and
attends over the filled part of the cache. It is functional in eager PyTorch
(returns a new pool), which keeps ``torch.export`` graphs pure; its TensorRT
translation is TensorRT Edge-LLM's ``AttentionPlugin`` (XQA decode kernels,
``export/tensorrt/edgellm_attention.py``), which writes the pool in place and
reads only the ``context_lengths`` filled slots instead of the whole cache.

The contract follows that plugin exactly (Edge-LLM 0.10.1):

  qkv                 (B, S, (Hq + 2*Hkv) * D)   q | k | v of the S new tokens
  kv_pool             (2, P, 128, Hkv, D)        K pages then V pages; viewed as
                                                 (2P, 128, Hkv, D), V page id = K page id + P
  context_lengths     (B,) int32                 tokens in the cache *after* this call
  rope_cos_sin        (1, max_pos, R) float32    cos(pos * inv_freq) | sin(...), R <= D
                                                 (rotate-half RoPE on the first R channels)
  kvcache_start_index (B,) int32                 cache slot of the first new token
  kv_page_table       (B, 2, M) int32            K page ids, then V page ids, per sequence
  -> attn (B, S, Hq, D), new kv_pool

New token i sits at position ``context_lengths - S + i``; it attends causally to
cache slots ``[0, position]``. Scores and softmax run in float32. Only B = 1 is
implemented eagerly (the framework's contract).
"""

import torch

from .base import AttentionKernel

KV_PAGE_SIZE = 128  # Edge-LLM rt::kTOKENS_PER_PAGE


def num_kv_pages(max_cache_len: int) -> int:
    return -(-int(max_cache_len) // KV_PAGE_SIZE)


def identity_page_table(num_pages: int) -> torch.Tensor:
    """(1, 2, P) int32: sequence 0 owns every page in order (V pages offset by P)."""
    pages = torch.arange(num_pages, dtype=torch.int32)
    return torch.stack([pages, pages + num_pages]).unsqueeze(0)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the first 2*cos.shape[-1] channels, float32 math (plugin kernel)."""
    half = cos.shape[-1]
    x32 = x.float()
    x1, x2, rest = x32[..., :half], x32[..., half:2 * half], x32[..., 2 * half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], dim=-1).to(x.dtype)


@torch.library.custom_op("tsurgeon::paged_attention", mutates_args=())
def paged_attention(
    qkv: torch.Tensor,
    kv_pool: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    kv_page_table: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    attention_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, _ = qkv.shape
    if batch != 1:
        raise NotImplementedError("tsurgeon::paged_attention: eager path supports batch 1 only.")
    hq, hkv, d = num_q_heads, num_kv_heads, head_size
    x = qkv.view(seq, hq + 2 * hkv, d)
    q, k, v = x[:, :hq], x[:, hq:hq + hkv], x[:, hq + hkv:]

    end = int(context_lengths[0])
    positions = torch.arange(end - seq, end, device=qkv.device)
    table = rope_cos_sin[0, positions].view(seq, 1, -1)
    half = table.shape[-1] // 2
    cos, sin = table[..., :half], table[..., half:]
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)

    pool = kv_pool.clone()
    num_pages = pool.shape[1]
    pages = pool.view(2 * num_pages, KV_PAGE_SIZE, hkv, d)
    k_pages = kv_page_table[0, 0].long()
    v_pages = kv_page_table[0, 1].long()
    pages[k_pages[positions // KV_PAGE_SIZE], positions % KV_PAGE_SIZE] = k
    pages[v_pages[positions // KV_PAGE_SIZE], positions % KV_PAGE_SIZE] = v

    used = -(-end // KV_PAGE_SIZE)
    keys = pages[k_pages[:used]].reshape(-1, hkv, d)[:end].float()     # (end, Hkv, D)
    values = pages[v_pages[:used]].reshape(-1, hkv, d)[:end].float()
    group = hq // hkv
    qg = q.float().view(seq, hkv, group, d)
    scores = torch.einsum("shgd,thd->hgst", qg, keys) * attention_scale  # (Hkv, G, S, end)
    causal = torch.arange(end, device=qkv.device)[None, :] > positions[:, None]
    scores = scores.masked_fill(causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hgst,thd->shgd", probs, values).reshape(1, seq, hq, d)
    return out.to(qkv.dtype), pool


@paged_attention.register_fake
def _(qkv, kv_pool, context_lengths, rope_cos_sin, kvcache_start_index, kv_page_table,
      num_q_heads, num_kv_heads, head_size, attention_scale):
    batch, seq, _ = qkv.shape
    return qkv.new_empty(batch, seq, num_q_heads, head_size), torch.empty_like(kv_pool)


class PagedKernel(AttentionKernel):
    """Fused paged attention (:func:`paged_attention`), used with the ``io_paged`` cache.

    Rotary embedding, cache write and attention over the filled slots are one
    operation; on TensorRT it maps to the Edge-LLM ``AttentionPlugin``.
    """

    name = "paged"
    requires_cache = "io_paged"
    fuses_projection = True

    def forward_fused(self, attn, x, in_seq_len, kv_pool, paged):
        """``paged`` = (context_lengths, rope_cos_sin, kvcache_start_index, kv_page_table)."""
        head_dim = attn.value_head_dim
        if (attn.rope_pruned or attn.key_head_dim != head_dim
                or attn.q_proj.out_features != attn.num_heads * head_dim):
            raise NotImplementedError(
                "cache_impl='io_paged' needs equal q/k/v head dims and unpruned RoPE (one cos/sin table "
                "for every head); use 'io_inplace' for within-head pruned attention.")
        qkv = torch.cat([attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)], dim=-1)
        qkv = qkv.reshape(1, in_seq_len, qkv.shape[-1])
        context_lengths, rope_cos_sin, start_index, page_table = paged
        attn_output, kv_pool = paged_attention(
            qkv, kv_pool, context_lengths, rope_cos_sin, start_index, page_table,
            attn.num_heads, attn.kv_num_heads, head_dim, float(attn.score_scale))
        out_shape = (1, in_seq_len, -1) if attn.add_batch_dim else (in_seq_len, -1)
        return attn.out_proj(attn_output.reshape(out_shape)), kv_pool, None


__all__ = ["KV_PAGE_SIZE", "identity_page_table", "num_kv_pages", "paged_attention", "PagedKernel"]
