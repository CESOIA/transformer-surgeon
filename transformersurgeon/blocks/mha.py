# Simple blocks implementation for transformers
# Multi Head Attention (MHA) blocks
import math
import torch
import torch.nn.functional as F
from .rope import apply_rope_multihead, build_rope_prune_projection
from . import LinearCompressed  

def attention(query, key, value, attn_mask=None):
    """
    Explicit implementation of scaled dot-product attention.
    This is needed for cases where torch's built-in SDPA is not suited for model export (e.g., ONNX).
    It supports GQA automatically by handling different head dimensions for query and key/value.
    key_cache and value_cache are provided separately to minimize the concatenation overhead. They are used only if is_causal is False.

    Args:
        query: Tensor of shape (seq_length, q_head_num, q_head_dim)
        key:   Tensor of shape (seq_length, kv_head_num, kv_head_dim)
        value: Tensor of shape (seq_length, kv_head_num, kv_head_dim)
        attn_mask: Optional boolean tensor of shape (1, 1, q_seq_length, kv_seq_length) where True indicates positions to mask (set to -inf in scores)        
    """
    _, q_head_num, head_dim = query.size()
    _, kv_head_num,       _ = key.size()
    group_size = q_head_num // kv_head_num

    dtype = query.dtype
    scale = 1.0 / math.sqrt(head_dim)

    # -------------------------------------------------------------------------
    # CURRENT PATH: materialize GQA expansion before matmul.
    # Avoids this block if switching to einsum or broadcasting alternatives below.
    # -------------------------------------------------------------------------
    # expand key, value for possible GQA
    # key   = key.repeat_interleave(group_size, dim=1)    # (seq_length, q_head_num, head_dim)
    # value = value.repeat_interleave(group_size, dim=1)  # (seq_length, q_head_num, head_dim)
    key   = key.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(-1, q_head_num, head_dim)
    value = value.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(-1, q_head_num, head_dim)

    # Swap head and sequence dimensions for attention computation
    query = query.transpose(0, 1)                        # (q_head_num, seq_len, head_dim)
    key   = key.transpose(0, 1)                          # (q_head_num, kv_len, head_dim)
    value = value.transpose(0, 1)                        # (q_head_num, kv_len, head_dim)

    scores = torch.matmul(query * scale, key.transpose(-2, -1)) # (q_head_num, seq_len, kv_len)

    if attn_mask is not None:
        scores = scores + attn_mask

    scores = torch.nn.functional.softmax(scores, dim=-1)

    attn_output = torch.matmul(scores.to(dtype), value)  # (q_head_num, seq_len, head_dim)
    # -------------------------------------------------------------------------
    # END CURRENT PATH
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ALTERNATIVE 1 — einsum GQA (no KV materialization).
    # To use: comment out the CURRENT PATH block above and uncomment this block.
    # -------------------------------------------------------------------------
    # query = query.transpose(0, 1).reshape(kv_head_num, group_size, -1, head_dim)  # (kv_heads, group, seq, head_dim)
    # key   = key.transpose(0, 1)    # (kv_heads, kv_len, head_dim)
    # value = value.transpose(0, 1)  # (kv_heads, kv_len, head_dim)
    #
    # scores = torch.einsum("hgsd,hkd->hgsk", query, key) * scale  # (kv_heads, group, seq, kv_len)
    #
    # if attn_mask is not None:
    #     scores = scores + attn_mask.unsqueeze(1)  # broadcast over group dim
    #
    # scores = torch.nn.functional.softmax(scores, dim=-1)
    #
    # attn_output = torch.einsum("hgsk,hkd->hgsd", scores.to(dtype), value)  # (kv_heads, group, seq, head_dim)
    # attn_output = attn_output.reshape(q_head_num, -1, head_dim)  # (q_heads, seq, head_dim)
    # -------------------------------------------------------------------------
    # END ALTERNATIVE 1
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ALTERNATIVE 2 — broadcasting GQA (expand without reshape, avoids copy).
    # To use: comment out the CURRENT PATH block above and uncomment this block.
    # -------------------------------------------------------------------------
    # query = query.transpose(0, 1).reshape(kv_head_num, group_size, -1, head_dim)  # (kv_heads, group, seq, head_dim)
    # key   = key.transpose(0, 1).unsqueeze(1)    # (kv_heads, 1, kv_len, head_dim)
    # value = value.transpose(0, 1).unsqueeze(1)  # (kv_heads, 1, kv_len, head_dim)
    #
    # scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (kv_heads, group, seq, kv_len) via broadcast
    #
    # if attn_mask is not None:
    #     scores = scores + attn_mask.unsqueeze(1)  # broadcast over group dim
    #
    # scores = torch.nn.functional.softmax(scores, dim=-1)
    #
    # attn_output = torch.matmul(scores.to(dtype), value)   # (kv_heads, group, seq, head_dim) via broadcast
    # attn_output = attn_output.reshape(q_head_num, -1, head_dim)  # (q_heads, seq, head_dim)
    # -------------------------------------------------------------------------
    # END ALTERNATIVE 2
    # -------------------------------------------------------------------------

    return attn_output.to(dtype)  # (seq_len, q_head_num, head_dim)

class MHABase(torch.nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            bias_required=None,
            kv_num_heads=None,
            use_sdpa=False,
            compression_config=None,
            dtype=None,
            **kwargs,
            ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert kv_num_heads is None or num_heads % kv_num_heads == 0, "num_heads must be divisible by kv_num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_num_heads = num_heads if kv_num_heads is None else kv_num_heads
        self.use_sdpa = use_sdpa
        self.kv_out_dim = self.kv_num_heads * self.head_dim
        self.dtype = dtype

        # Setup compression configuration
        if compression_config is None:
            compression_config = {
                "q_proj": {"lrd": {"rank": "full"}},
                "k_proj": {"lrd": {"rank": "full"}},
                "v_proj": {"lrd": {"rank": "full"}},
                "out_proj": {"lrd": {"rank": "full"}}
            }

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "q_proj": False,
                "k_proj": False,
                "v_proj": False,
                "out_proj": False
            }
        else:
            bias_required = dict(bias_required)
            bias_required.setdefault("q_proj", False)
            bias_required.setdefault("k_proj", False)
            bias_required.setdefault("v_proj", False)
            bias_required.setdefault("out_proj", False)
        
        q_lrd_rank = compression_config["q_proj"]["lrd"]["rank"]
        k_lrd_rank = compression_config["k_proj"]["lrd"]["rank"]
        v_lrd_rank = compression_config["v_proj"]["lrd"]["rank"]
        out_lrd_rank = compression_config["out_proj"]["lrd"]["rank"]
        # Structured-pruning-aware sizing for attention, partially resolved:
        # q_proj/k_proj may now be hard-pruned (their head_dim shrinks) -- forward()
        # below reads their *current* out_features rather than a fixed head_dim, and
        # _project_rope projects RoPE's cos/sin to match when they're position_linked.
        # Still TODO: v_proj. Pruning it shrinks attn_output's last dim, and under GQA
        # the v->o coupling is not 1:1 (repeat_interleave duplicates each kv head's
        # slice group_size times before o_proj, so CoupledPruner's plain index-select
        # on o_proj's input columns -- sized off v_proj's un-expanded mask -- doesn't
        # line up); out_proj's in_features and the reshape after attention are not
        # adjusted for a pruned v_proj.

        self.q_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["q_proj"],
            rank=q_lrd_rank,
            dtype=dtype)
        self.k_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["k_proj"],
            rank=k_lrd_rank,
            dtype=dtype)
        self.v_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["v_proj"],
            rank=v_lrd_rank,
            dtype=dtype)
        self.out_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["out_proj"],
            rank=out_lrd_rank,
            dtype=dtype)

        # Cached RoPE-frequency selection matrix for pruned q/k (see
        # _project_rope below). None until (and unless) q_proj/k_proj are
        # hard-pruned; rebuilt lazily the first time it's needed.
        self.register_buffer("rope_freq_proj", None, persistent=False)

    def _project_rope(self, cos, sin):
        """Project precomputed RoPE ``cos``/``sin`` down to the pruned rotary
        dimension, when structured pruning has hard-pruned q_proj/k_proj.

        Structured pruning of a position-linked q_proj/k_proj (see
        ``pruning.position_linked`` in the model's indexing) only removes
        output rows -- it does not know which rotary frequency each surviving
        row corresponds to. ``cos``/``sin`` are still precomputed at the
        original (unpruned) head_dim, so they must be re-indexed with the
        exact same keep pattern before being applied to the now-narrower
        q/k, via a small static 0/1 projection matrix built from the
        pruning mask (``build_rope_prune_projection`` in ``blocks/rope.py``).

        No-op (returns cos/sin unchanged) if q_proj/k_proj are not pruned.
        Raises ``ValueError`` if only one of q_proj/k_proj was pruned, since
        the pair must always be pruned together for RoPE to remain valid.
        """
        q_head_dim = self.q_proj.out_features // self.num_heads
        k_head_dim = self.k_proj.out_features // self.kv_num_heads
        q_pruned = q_head_dim < self.head_dim
        k_pruned = k_head_dim < self.head_dim

        if not q_pruned and not k_pruned:
            return cos, sin

        if q_pruned != k_pruned:
            raise ValueError(
                "Structured pruning was applied to only one of q_proj/k_proj "
                f"(head_dim: q_proj={q_head_dim}, k_proj={k_head_dim}, original="
                f"{self.head_dim}). RoPE requires q_proj and k_proj to be pruned "
                "together (they are coupled via pruning.position_linked / "
                "pruning.coupled_masks) -- prune both with a shared mask, or "
                "neither."
            )

        if q_head_dim != k_head_dim:
            raise ValueError(
                f"q_proj and k_proj were pruned to different head dims "
                f"({q_head_dim} vs {k_head_dim}); RoPE requires them to match."
            )

        proj = self.rope_freq_proj
        if proj is None or proj.shape[0] != q_head_dim // 2:
            q_mask = getattr(self.q_proj, "rope_prune_mask", None)
            k_mask = getattr(self.k_proj, "rope_prune_mask", None)
            if q_mask is None or k_mask is None:
                raise ValueError(
                    "q_proj/k_proj head_dim has shrunk but no rope_prune_mask was "
                    "recorded by structured pruning; cannot determine which rotary "
                    "frequencies survived. This should not happen if pruning went "
                    "through StructuredPruner on a position_linked scheme."
                )
            proj = build_rope_prune_projection(
                q_mask, k_mask, self.head_dim, self.num_heads, self.kv_num_heads,
            )
            self.rope_freq_proj = proj.to(device=cos.device, dtype=cos.dtype)
            proj = self.rope_freq_proj

        proj = proj.to(device=cos.device, dtype=cos.dtype)
        return cos @ proj.t(), sin @ proj.t()

class MHAEncoder(MHABase): # No cache, no causal masking, for encoder-only use
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, rope=None):
        seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V. q_proj/k_proj's per-head dim may be smaller
        # than self.head_dim if position-linked structured pruning hard-pruned
        # them (see _project_rope) -- v_proj is uncoupled from that pruning and
        # always keeps self.head_dim.
        q = self.q_proj(x).view(seq_length, self.num_heads, self.q_proj.out_features // self.num_heads) # (seq_length, num_heads, q_head_dim)
        k = self.k_proj(x).view(seq_length, self.kv_num_heads, self.k_proj.out_features // self.kv_num_heads) # (seq_length, kv_num_heads, k_head_dim)
        v = self.v_proj(x).view(seq_length, self.kv_num_heads, self.head_dim) # (seq_length, kv_num_heads, head_dim)

        # Apply RoPE (expects (seq_length, num_heads, head_dim), not transposed)
        if rope is not None:
            cos, sin = rope
            cos, sin = self._project_rope(cos, sin)
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)

        # Scaled dot-product attention. `attention()` takes (seq, heads, dim) and
        # transposes internally, same as `q`/`k`/`v` above -- only the SDPA path
        # needs the (heads, seq, dim) layout, so transpose just for that call.
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                is_causal=False, enable_gqa=True,
            )
        else:
            attn_output = attention(q, k, v)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, embed_dim)
        output = self.out_proj(attn_output)
        
        return output

class MHAEncoderFusedProj(torch.nn.Module): # Qwen-style fused projection MHA (No GQA)
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias_required=None,
        use_sdpa=False,
        compression_config=None,
        ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sdpa = use_sdpa

        # Setup compression configuration
        if compression_config is None:
            compression_config = {
                "qkv_proj": {"lrd": {"rank": "full"}},
                "out_proj": {"lrd": {"rank": "full"}}
            }

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "qkv_proj": False,
                "out_proj": False
            }
        else:
            bias_required = dict(bias_required)
            bias_required.setdefault("qkv_proj", False)
            bias_required.setdefault("out_proj", False)
        
        qkv_lrd_rank = compression_config["qkv_proj"]["lrd"]["rank"]
        out_lrd_rank = compression_config["out_proj"]["lrd"]["rank"]
        
        # Instantiate layers
        self.qkv_proj = LinearCompressed(
            embed_dim, 
            3 * embed_dim, 
            bias=bias_required["qkv_proj"],
            rank=qkv_lrd_rank)
        self.out_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["out_proj"],
            rank=out_lrd_rank)
        
    def forward(self, x, rope=None):
        seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V in a single projection
        qkv = self.qkv_proj(x).view(seq_length, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)
    
        # Scaled dot-product attention
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=False)
        else:
            attn_output = attention(q, k, v)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(1, 0, 2).view(seq_length, embed_dim).contiguous()
        output = self.out_proj(attn_output)
        
        return output
    
CACHE_IMPLS = ("mutable", "io_scatter", "io_concat")


class MHACausal(MHABase): # Causal MHA with caching for decoder use
    """Causal multi-head attention over a fixed-size KV cache.

    The cache write is selected by ``cache_impl`` (see ``CACHE_IMPLS``):

    - ``"mutable"``   : in-place ``index_copy_`` on an internal buffer. The cache
                        is module state (not graph I/O). QNN-only; the current
                        default so existing exports/tests are unchanged.
    - ``"io_scatter"``: functional ``index_put`` on a cache passed in as a
                        forward argument and returned as output. Portable to
                        QNN (ScatterNd) and TensorRT (ScatterND).
    - ``"io_concat"`` : scatter-free functional write built from a positional
                        mask + ``where`` (Concat/select family). Universal
                        fallback for backends lacking an index_put converter.

    All three produce an updated **fixed-size** ``(max_cache_len, kv_num_heads,
    head_dim)`` cache and then run the identical mask + attention code, so the
    three paths are numerically equivalent. The framework feeds one token at a
    time (``in_seq_len == 1``); the writers assume that.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache_impl = kwargs.get("cache_impl", "mutable")
        if self.cache_impl not in CACHE_IMPLS:
            raise ValueError(
                f"Unsupported cache_impl {self.cache_impl!r}; expected one of {CACHE_IMPLS}"
            )

        # Initialize key and value caches
        self.max_cache_length=kwargs.get("max_cache_len", 2048)

        # Buffers back the "mutable" path; harmless (and cheap) for io_* modes,
        # where they also serve as the initial cache when none is passed in.
        self.register_buffer(
            "key_cache",
            torch.zeros(
                self.max_cache_length,
                self.kv_num_heads,
                self.head_dim,
                dtype=self.dtype),
            persistent=False
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                self.max_cache_length,
                self.kv_num_heads,
                self.head_dim,
                dtype=self.dtype),
            persistent=False
        )

    def _write_mutable(self, k, v, write_index, key_cache, value_cache):
        """In-place write on the internal buffers (QNN peak path)."""
        with torch.no_grad():
            self.key_cache.index_copy_(0, write_index, k)
            self.value_cache.index_copy_(0, write_index, v)
        return self.key_cache, self.value_cache

    def _write_scatter(self, k, v, write_index, key_cache, value_cache):
        """Functional scatter -> updated fixed-size cache (aten.index_put)."""
        key_cache = key_cache.index_put((write_index,), k)
        value_cache = value_cache.index_put((write_index,), v)
        return key_cache, value_cache

    def _write_concat(self, k, v, write_index, key_cache, value_cache):
        """Scatter-free write: positional one-hot mask + where (in_seq_len==1)."""
        idx = torch.arange(self.max_cache_length, device=k.device)
        sel = (idx == write_index).view(self.max_cache_length, 1, 1)  # (L,1,1)
        k_row = k.reshape(1, self.kv_num_heads, self.head_dim)
        v_row = v.reshape(1, self.kv_num_heads, self.head_dim)
        key_cache = torch.where(sel, k_row, key_cache)
        value_cache = torch.where(sel, v_row, value_cache)
        return key_cache, value_cache

    def forward(self, x, pos_id, pos_id_list, mask_penalty,
                key_cache=None, value_cache=None, rope=None):
        """
        Forward pass of the causal Multi-Head Attention with caching.

        Args:
            x (torch.Tensor): Input tensor of shape (in_seq_len, embed_dim).
            pos_id (int): Current length of the cache (number of tokens already in cache). This is used to determine where to write the new keys and values in the cache.
            key_cache/value_cache (torch.Tensor, optional): Incoming fixed-size
                caches for the ``io_*`` modes. If omitted, the internal buffers
                are used as the initial cache.
            rope (tuple, optional): Tuple of (cos, sin) tensors for RoPE application.

        Returns:
            output (mutable mode) or (output, key_cache, value_cache) (io modes).
        """
        in_seq_len, embed_dim = x.size()

        # Fall back to internal buffers when caches are not threaded in.
        if key_cache is None:
            key_cache = self.key_cache
        if value_cache is None:
            value_cache = self.value_cache

        # Project inputs to Q, K, V. q_proj/k_proj's per-head dim may be smaller
        # than self.head_dim if position-linked structured pruning hard-pruned
        # them (see _project_rope) -- v_proj is uncoupled from that pruning and
        # always keeps self.head_dim. NOTE: key_cache/value_cache below are
        # still allocated at self.head_dim (attention resizing for the causal
        # KV-cache is a separate, not-yet-supported piece of work -- see the
        # TODO in MHABase.__init__), so a hard-pruned k_proj will fail at the
        # cache write below, not silently produce wrong results.
        q = self.q_proj(x).view(in_seq_len, self.num_heads, self.q_proj.out_features // self.num_heads) # (in_seq_len, num_heads, q_head_dim)
        k = self.k_proj(x).view(in_seq_len, self.kv_num_heads, self.k_proj.out_features // self.kv_num_heads) # (in_seq_len, kv_num_heads, k_head_dim)
        v = self.v_proj(x).view(in_seq_len, self.kv_num_heads, self.head_dim) # (in_seq_len, kv_num_heads, head_dim)

        # Apply RoPE
        if rope is not None:
            cos = rope[0][pos_id]
            sin = rope[1][pos_id]
            cos, sin = self._project_rope(cos, sin)
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)

        # Write new K/V into the fixed-size cache (mechanism per cache_impl).
        write_index = torch.clamp(pos_id, 0, self.max_cache_length-1).long()
        if self.cache_impl == "mutable":
            key_cache, value_cache = self._write_mutable(k, v, write_index, key_cache, value_cache)
        elif self.cache_impl == "io_scatter":
            key_cache, value_cache = self._write_scatter(k, v, write_index, key_cache, value_cache)
        else:  # io_concat
            key_cache, value_cache = self._write_concat(k, v, write_index, key_cache, value_cache)

        # On-the-fly attention mask
        q_pos, k_pos = pos_id_list
        attn_mask = torch.where((q_pos < k_pos), mask_penalty, torch.zeros_like(mask_penalty))[pos_id].unsqueeze(0) # (1, max_cache_len)

        # Call mha operation (SDPA or custom attention)
        if self.use_sdpa:
            q = q.transpose(0, 1) # (num_heads, in_seq_len, head_dim)
            k = key_cache.transpose(0, 1) # (kv_num_heads, pos_id, head_dim)
            v = value_cache.transpose(0, 1) # (kv_num_heads, pos_id, head_dim)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True)
        else:
            attn_output = attention(q, key_cache, value_cache, attn_mask)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(0, 1).reshape(in_seq_len, embed_dim)
        output = self.out_proj(attn_output)

        if self.cache_impl == "mutable":
            return output
        return output, key_cache, value_cache

__all__ = [
    "MHAEncoder",
    "MHACausal",
    "MHAEncoderFusedProj",
    "CACHE_IMPLS",
    ]
