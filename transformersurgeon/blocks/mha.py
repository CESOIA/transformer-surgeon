# Simple blocks implementation for transformers
# Multi Head Attention (MHA) blocks
import math
import torch
import torch.nn.functional as F
from .rope import apply_rope_multihead
from . import LinearCompressed

def attention_mask(
    q_seq_length,
    kv_seq_length,          # effective KV length (current cache_len)
    device="cpu",
    kv_alloc_length=None,   # optional fixed width for static export
    mask_value=-1e6,
):
    # If kv_alloc_length is set, mask shape is fixed to that width.
    # Otherwise dynamic mode uses kv_seq_length as width.
    kv_width = kv_seq_length if kv_alloc_length is None else kv_alloc_length

    i = torch.arange(q_seq_length, device=device).view(1, 1, q_seq_length, 1)
    j = torch.arange(kv_width, device=device).view(1, 1, 1, kv_width)

    # Align each query row to absolute position based on effective cache length.
    q_abs_pos = i + (kv_seq_length - q_seq_length)

    # Standard causal future mask
    future = j > q_abs_pos

    # In static mode, also mask slots not written yet (j >= kv_seq_length)
    not_filled = j >= kv_seq_length

    return (future | not_filled).to(torch.float32) * mask_value

def attention(query, key, value, attn_mask=None):
    """
    Explicit implementation of scaled dot-product attention.
    This is needed for cases where torch's built-in SDPA is not suited for model export (e.g., ONNX).
    It supports GQA automatically by handling different head dimensions for query and key/value.
    key_cache and value_cache are provided separately to minimize the concatenation overhead. They are used only if is_causal is False.

    Args:
        query: Tensor of shape (batch_size, seq_length, q_head_num, q_head_dim)
        key:   Tensor of shape (batch_size, seq_length, kv_head_num, kv_head_dim)
        value: Tensor of shape (batch_size, seq_length, kv_head_num, kv_head_dim)
        attn_mask: Optional boolean tensor of shape (1, 1, q_seq_length, kv_seq_length) where True indicates positions to mask (set to -inf in scores)        
    """
    _, _, q_head_num, head_dim = query.size()
    _, _, kv_head_num,       _ = key.size()
    group_size = q_head_num // kv_head_num

    # expand key, value for possible GQA
    key   = key.repeat_interleave(group_size, dim=2)    # (batch_size, seq_length, q_head_num, head_dim)
    value = value.repeat_interleave(group_size, dim=2)  # (batch_size, seq_length, q_head_num, head_dim)

    dtype = query.dtype

    # Swap head and sequence dimensions for attention computation
    query = query.transpose(1, 2)
    key   = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Scaled QK^T
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1))*scale

    # Generate negative causal mask
    if attn_mask is not None:
        # Mask the future positions
        scores = scores + attn_mask # Add large negative values for masking

    # Softmax
    scores = torch.nn.functional.softmax(scores.to(torch.float32), dim=-1)

    # Project the values over the scores
    attn_output = torch.matmul(scores.to(dtype), value)  # (batch_size, kv_head_num, group_size, seq_length, head_dim)

    return attn_output.to(dtype)

class MHABase(torch.nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            bias_required=None,
            kv_num_heads=None,
            use_sdpa=False,
            compression_config=None,
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
        
        self.q_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["q_proj"],
            rank=q_lrd_rank)
        self.k_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["k_proj"],
            rank=k_lrd_rank)
        self.v_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["v_proj"],
            rank=v_lrd_rank)
        self.out_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["out_proj"],
            rank=out_lrd_rank)

class MHAEncoder(MHABase): # No cache, no causal masking, for encoder-only use
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, rope=None):
        batch_size, seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_length, head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.kv_num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, kv_num_heads, seq_length, head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.kv_num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, kv_num_heads, seq_length, head_dim)

        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)
        
        # Scaled dot-product attention
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        else:
            attn_output = attention(q, k, v)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).view(batch_size, seq_length, embed_dim).contiguous()
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
        batch_size, seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V in a single projection
        qkv = self.qkv_proj(x).view(batch_size, seq_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)
    
        # Scaled dot-product attention
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        else:
            attn_output = attention(q, k, v)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).view(batch_size, seq_length, embed_dim).contiguous()
        output = self.out_proj(attn_output)
        
        return output
    
class MHACausal(MHABase): # Causal MHA with caching for decoder use
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_kvcache(
            batch_size=kwargs.get("batch_size", 1),
            max_cache_length=kwargs.get("max_cache_len", 2048)
        )

    def _init_kvcache(self, batch_size, max_cache_length):
        # Initialize key and value caches as buffers (not parameters, not persistent)
        self.register_buffer("key_cache", torch.zeros(batch_size, max_cache_length, self.kv_num_heads, self.head_dim), persistent=False)
        self.register_buffer("value_cache", torch.zeros(batch_size, max_cache_length, self.kv_num_heads, self.head_dim), persistent=False)
        
    def forward(self, x, last_pos, attn_mask=None, rope=None, static=False):
        """
        Forward pass of the causal Multi-Head Attention with caching.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_seq_len, embed_dim).
            last_pos (int): Current length of the cache (number of tokens already in cache). This is used to determine where to write the new keys and values in the cache.
            rope (tuple, optional): Tuple of (cos, sin) tensors for RoPE application.
        """
        batch_size, in_seq_len, embed_dim = x.size()

        # Project inputs to Q, K, V
        q = self.q_proj(x).view(batch_size, in_seq_len, self.num_heads, self.head_dim) # (batch_size, in_seq_len, num_heads, head_dim)
        k = self.k_proj(x).view(batch_size, in_seq_len, self.kv_num_heads, self.head_dim) # (batch_size, in_seq_len, kv_num_heads, head_dim)
        v = self.v_proj(x).view(batch_size, in_seq_len, self.kv_num_heads, self.head_dim) # (batch_size, in_seq_len, kv_num_heads, head_dim)
        
        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)

        # Update key and value caches in-place
        with torch.no_grad():
            if static:
                last_pos_tensor = torch.as_tensor(last_pos, device=x.device)
                indexes = torch.arange(-in_seq_len, 0, device=x.device) + last_pos_tensor
            else:
                indexes = torch.arange(last_pos-in_seq_len, last_pos, device=x.device)
            self.key_cache.index_copy_(1, indexes, k)
            self.value_cache.index_copy_(1, indexes, v)

        # Slice cache
        if static:
            key_cache = self.key_cache
            value_cache = self.value_cache
        else:
            key_cache = self.key_cache[:, :last_pos] # (batch_size, last_pos, kv_num_heads, head_dim)
            value_cache = self.value_cache[:, :last_pos] # (batch_size, last_pos, kv_num_heads, head_dim)

        if self.use_sdpa:
            q = q.transpose(1, 2) # (batch_size, num_heads, in_seq_len, head_dim)
            k = key_cache.transpose(1, 2) # (batch_size, kv_num_heads, last_pos, head_dim)
            v = value_cache.transpose(1, 2) # (batch_size, kv_num_heads, last_pos, head_dim)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True)
        else:
            attn_output = attention(q, key_cache, value_cache, attn_mask)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, in_seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output
    
__all__ = [
    "attention_mask",
    "MHAEncoder",
    "MHACausal",
    "MHAEncoderFusedProj"
    ]
