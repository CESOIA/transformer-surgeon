# Simple blocks implementation for transformers
# Multi Head Attention (MHA) blocks
import math
import torch
import torch.nn.functional as F
from . import apply_rope_multihead

def attention(query, key, value, is_causal=False):
    """
    Explicit implementation of scaled dot-product attention.
    This is needed for cases where torch's built-in SDPA is not suited for model export (e.g., ONNX).

    Args:
        query: Tensor of shape (batch_size, num_heads, seq_length, head_dim)
        key:   Tensor of shape (batch_size, num_heads, seq_length, head_dim)
        value: Tensor of shape (batch_size, num_heads, seq_length, head_dim)
        is_causal: Whether to apply causal masking (for decoder use)
    """
    _, _, seq_length, head_dim = query.size()
    device = query.device
    dtype = query.dtype
        
    # Scaled QK^T
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.permute(0, 1, 3, 2))*scale
    # Cast to float32 for numerical stability
    scores = scores.to(torch.float32)
    if is_causal:
        # Generate negative causal mask
        i = torch.arange(seq_length, device=device).view(1,1,seq_length,1)
        j = torch.arange(seq_length, device=device).view(1,1,1,seq_length)
        mask = i < j
        # Mask the future positions
        scores = scores.masked_fill(mask, torch.finfo(torch.float32).min)
    # Stabilize softmax by subtracting max
    scores = scores - scores.max(dim=-1, keepdim=True).values
    # Softmax and back to original dtype
    scores = torch.nn.functional.softmax(scores, dim=-1).to(dtype)
    # Attention output
    attn_output = torch.matmul(scores, value)

    return attn_output

class MHAEncoder(torch.nn.Module): # No cache, no causal masking, for encoder-only use
    def __init__(self, embed_dim, num_heads, bias=False, use_sdpa=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sdpa = use_sdpa
        
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(self, x, rope=None):
        batch_size, seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)
        
        # Scaled dot-product attention
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_output = attention(q, k, v, is_causal=False)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim).contiguous()
        output = self.out_proj(attn_output)
        
        return output

class MHAEncoderFusedProj(torch.nn.Module): # Qwen-style fused projection MHA
    def __init__(self, embed_dim, num_heads, bias=False, use_sdpa=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sdpa = use_sdpa
        
        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(self, x, rope=None):
        batch_size, seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V in a single projection
        qkv = self.qkv_proj(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)
    
        # Scaled dot-product attention
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_output = attention(q, k, v, is_causal=False)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim).contiguous()
        output = self.out_proj(attn_output)
        
        return output
    
class MHACausal(torch.nn.Module): # Causal MHA with caching for decoder use
    def __init__(self, embed_dim, num_heads, bias=False, use_sdpa=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sdpa = use_sdpa
        
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(self, x, key_cache=None, value_cache=None, prefill=True, rope=None):
        batch_size, seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)

        # Append to cache
        if not prefill:
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        
        # Scaled dot-product attention
        if self.use_sdpa:
            if prefill:
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            if prefill:
                attn_output = attention(q, k, v, is_causal=True)
            else:
                attn_output = attention(q, k, v, is_causal=False)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim).contiguous()
        output = self.out_proj(attn_output)
        
        return output, k, v
    
__all__ = [
    "MHAEncoder",
    "MHACausal",
    "MHAEncoderFusedProj"
    ]