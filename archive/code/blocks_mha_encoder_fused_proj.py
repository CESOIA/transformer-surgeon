"""ARCHIVED: MHAEncoderFusedProj from blocks/mha.py.

Removed during the 2026-10-28 refactor: no model dictionary selected it
(``attn_type="mha_encoder_fused_proj"`` appeared nowhere), no test covered it,
and its tensor layouts did not match what ``attention()`` and
``apply_rope_multihead`` expect. A fused query/key/value attention atom would
also need a weight-splitting rule in the converter. Kept for reference only.
"""
import torch

from transformersurgeon.blocks.layers import LinearCompressed
from transformersurgeon.blocks.attention.kernels.manual import attention
from transformersurgeon.blocks.positions.rope import apply_rope_multihead


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
