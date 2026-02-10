# Simple blocks implementation for transformers
# Multi Head Attention (MHA) blocks
import math
import torch
import torch.nn.functional as F
from .rope import apply_rope_multihead
from ..layers import LinearCompressed

def attention(query, key, value, is_causal=False):
    """
    Explicit implementation of scaled dot-product attention.
    This is needed for cases where torch's built-in SDPA is not suited for model export (e.g., ONNX).
    It supports GQA automatically by handling different head dimensions for query and key/value.
    key_cache and value_cache are provided separately to minimize the concatenation overhead. They are used only if is_causal is False.

    Args:
        query: Tensor of shape (batch_size, num_heads, seq_length, q_head_dim)
        key:   Tensor of shape (batch_size, num_heads, seq_length, kv_head_dim)
        value: Tensor of shape (batch_size, num_heads, seq_length, kv_head_dim)
        key_cache [optional]: Tensor of shape (batch_size, num_heads, cache_seq_length, kv_head_dim)
        value_cache [optional]: Tensor of shape (batch_size, num_heads, cache_seq_length, kv_head_dim)
        is_causal: Whether to apply causal masking (for decoder use)
    """
    batch_size, q_head_num,  q_seq_length,  head_dim = query.size()
    _,          kv_head_num, kv_seq_length, _        = key.size()
    group_size = q_head_num // kv_head_num
    
    # reshape query, key, value for possible GQA
    query = query.view(batch_size, kv_head_num, group_size, q_seq_length, head_dim) # (batch_size, kv_head_num, group_size, q_seq_length, head_dim)
    key   = key.unsqueeze(2)    # (batch_size, kv_head_num, 1, kv_seq_length, head_dim)
    value = value.unsqueeze(2)  # (batch_size, kv_head_num, 1, kv_seq_length, head_dim)

    device = query.device
    dtype = query.dtype

    # Cast to float32 for numerical stability - avoid overflow/underflow
    query = query.to(torch.float32)
    key   = key.to(torch.float32)
    value = value.to(torch.float32)

    # Scaled QK^T
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.permute(0, 1, 2, 4, 3))*scale    

    # Generate negative causal mask
    if is_causal:
        i = torch.arange(q_seq_length, device=device).view(1, 1, q_seq_length, 1)
        j = torch.arange(kv_seq_length, device=device).view(1, 1, 1, kv_seq_length)
        i = i + (kv_seq_length - q_seq_length)  # Adjust query indexing to match key indexing in cache scenario
        mask = i < j
        # Mask the future positions
        scores = scores.masked_fill(mask, -1e4) # Use a large negative value for masking

    # Stabilize softmax by subtracting max
    scores = scores - scores.max(dim=-1, keepdim=True).values

    # Softmax
    scores = torch.nn.functional.softmax(scores, dim=-1)

    # Project the values over the scores
    attn_output = torch.matmul(scores, value)  # (batch_size, kv_head_num, group_size, seq_length, head_dim)

    # reshape back to original shape and dtype
    attn_output = attn_output.view(batch_size, q_head_num, q_seq_length, head_dim).to(dtype) # (batch_size, num_heads, seq_length, head_dim)

    return attn_output

class MHABase(torch.nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            bias_required=None,
            kv_num_heads=None,
            use_sdpa=False,
            compression_config=None,
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
                "q_proj": {"lrd_rank": "full"},
                "k_proj": {"lrd_rank": "full"},
                "v_proj": {"lrd_rank": "full"},
                "out_proj": {"lrd_rank": "full"}
            }

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "q_proj": False,
                "k_proj": False,
                "v_proj": False,
                "out_proj": False
            }
        
        q_lrd_rank = compression_config["q_proj"]["lrd_rank"]
        k_lrd_rank = compression_config["k_proj"]["lrd_rank"]
        v_lrd_rank = compression_config["v_proj"]["lrd_rank"]
        out_lrd_rank = compression_config["out_proj"]["lrd_rank"]
        
        self.q_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["q_proj"],
            lrd_rank=q_lrd_rank)
        self.k_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["k_proj"],
            lrd_rank=k_lrd_rank)
        self.v_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["v_proj"],
            lrd_rank=v_lrd_rank)
        self.out_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["out_proj"],
            lrd_rank=out_lrd_rank)

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
            attn_output = attention(q, k, v, is_causal=False)
        
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
                "qkv_proj": {"lrd_rank": "full"},
                "out_proj": {"lrd_rank": "full"}
            }

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "qkv_proj": False,
                "out_proj": False
            }
        
        qkv_lrd_rank = compression_config["qkv_proj"]["lrd_rank"]
        out_lrd_rank = compression_config["out_proj"]["lrd_rank"]
        
        # Instantiate layers
        self.qkv_proj = LinearCompressed(
            embed_dim, 
            3 * embed_dim, 
            bias=bias_required["qkv_proj"],
            lrd_rank=qkv_lrd_rank)
        self.out_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["out_proj"],
            lrd_rank=out_lrd_rank)
        
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
            attn_output = attention(q, k, v, is_causal=False)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).view(batch_size, seq_length, embed_dim).contiguous()
        output = self.out_proj(attn_output)
        
        return output
    
class MHACausal(MHABase): # Causal MHA with caching for decoder use
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    # def forward(self, x, key_cache=None, value_cache=None, prefill=True, rope=None):
    def forward(self, x, key_cache, value_cache, rope=None):
        """
        Forward pass of the causal Multi-Head Attention with caching.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
            key_cache (torch.Tensor): Key cache of shape (batch_size, num_heads, cache_length, head_dim). If None, no cache is used.
            value_cache (torch.Tensor, optional): Value cache of shape (batch_size, num_heads, cache_length, head_dim). If None, no cache is used.
            prefill (bool): If True, indicates that we are in prefill mode (processing multiple tokens). If False, we are decoding a single token.
            rope (tuple, optional): Tuple of (cos, sin) tensors for RoPE application.
        """
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
        
        # Concatenate caches if not in prefill mode
        # if not prefill:
            # k = torch.cat([key_cache, k], dim=2)   # (batch_size, num_heads, cache_length + seq_length, head_dim)
            # v = torch.cat([value_cache, v], dim=2) # (batch_size, num_heads, cache_length + seq_length, head_dim)
        key_cache = torch.cat([key_cache, k], dim=2)   # (batch_size, num_heads, 1 + cache_length + seq_length, head_dim)
        value_cache = torch.cat([value_cache, v], dim=2) # (batch_size, num_heads, 1 + cache_length + seq_length, head_dim)
        # Remove dummy cache entry (necessary for prefill phase)
        k = key_cache[:, :, 1:, :]
        v = value_cache[:, :, 1:, :]

        # Scaled dot-product attention
        # if self.use_sdpa:
        #     if prefill:
        #         attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        #     else:
        #         attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        # else:
        #     if prefill:
        #         attn_output = attention(q, k, v, is_causal=True)
        #     else:
        #         attn_output = attention(q, k, v, is_causal=False)
        # if self.use_sdpa:
        #     attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        # else:
        #     attn_output = attention(q, k, v, is_causal=True)
        attn_output = attention(q, k, v, is_causal=True)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        
        return output, key_cache, value_cache
    
__all__ = [
    "MHAEncoder",
    "MHACausal",
    "MHAEncoderFusedProj"
    ]
