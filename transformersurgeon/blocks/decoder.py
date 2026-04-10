import torch
import math
from typing import Tuple
from .mha import MHACausal, attention_mask
from .mlp import MLP
from .mlp import MLPGated
from .norm import RMSNorm
from .rope import (
    precompute_rope_inv_freqs,
    precompute_rope_cos_sin_half,
)
from .atomic import AtomicSum

class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, config : dict):
        super().__init__()
        """
        Generic Transformer Decoder Block.
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                - embed_dim (int): Embedding dimension.
                - num_heads (int): Number of attention heads.
                - mlp_hidden_dim (int): Hidden dimension of the MLP.
                - mlp_activation (str): Activation function for the MLP.
                - kv_num_heads (int, optional): Number of key/value heads for GQA.
                - gated_mlp (bool, optional): Whether to use gated MLP (Qwen-style).
                - compression_config (dict, optional): Compression configuration for MHA and MLP layers
        """

        # Extract configuration (strict schema).
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.mlp_hidden_dim = config["mlp_hidden_dim"]
        self.mlp_activation = config["mlp_activation"]
        self.mha_type = config["mha_type"]
        self.mlp_type = config["mlp_type"]
        self.norm_type = config["norm_type"]
        self.use_sdpa = config["use_sdpa"]
        self.max_cache_len = config["max_cache_len"]

        # Extract configuration (optional)
        self.kv_num_heads = None if "kv_num_heads" not in config else config["kv_num_heads"]
        self.compression_config = {
            "attn": {},
            "mlp": {},
        } if "compression_config" not in config else config["compression_config"]
        self.bias_required = {
            "attn": {},
            "mlp": {},
        } if "bias_required" not in config else config["bias_required"]

        # Instantiate normalization modules
        if self.norm_type == "rmsnorm":
            self.norm_in = RMSNorm(self.embed_dim)
            self.norm_out = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")

        # Instantiate attention module
        if self.mha_type == "mha_causal":
            self.attn = MHACausal(
                self.embed_dim, # QUERY/KEY, VALUE, OUTPUT DIMENSIONS MUST BE EXPLICITED
                self.num_heads,
                bias_required=self.bias_required["attn"],
                kv_num_heads=self.kv_num_heads,
                compression_config=self.compression_config["attn"],
                use_sdpa=self.use_sdpa,
                max_cache_len=self.max_cache_len,)
        else:
            raise ValueError(f"Unsupported MHA type: {self.mha_type}")      
        
        # Instantiate multi-layer perceptron module
        if self.mlp_type == "mlp_gated":
            self.mlp = MLPGated(
                self.embed_dim,
                self.mlp_hidden_dim,
                bias_required=self.bias_required["mlp"], 
                activation=self.mlp_activation,
                compression_config=self.compression_config["mlp"])
        elif self.mlp_type == "mlp":
            self.mlp = MLP(
                self.embed_dim,
                self.mlp_hidden_dim,
                bias_required=self.bias_required["mlp"],
                activation=self.mlp_activation,
                compression_config=self.compression_config["mlp"])
        else:
            raise ValueError(f"Unsupported MLP type: {self.mlp_type}")
        
        # Instantiate skip connection atomicity module
        self.atomic_sum = AtomicSum()

    def forward(
            self,
            x,
            cache_len=None,
            attn_mask=None,
            rope=None,
    ):
        """
        Forward pass of the Transformer Decoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_seq_len, embed_dim).
            key_cache (torch.Tensor): Key cache for attention of size (batch_size, num_heads, seq_len+1, head_dim). # Modified in-place.
            value_cache (torch.Tensor): Value cache for attention of size (batch_size, num_heads, seq_len+1, head_dim). # Modified in-place.
            rope (tuple(torch.Tensor, torch.Tensor), optional): Precomputed RoPE cos/sin for queries and keys of size (1, 1, seq_len, head_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            torch.Tensor: Updated key cache of shape (batch_size, num_heads, total_seq_len+1, head_dim).
            torch.Tensor: Updated value cache of shape (batch_size, num_heads, total_seq_len+1, head_dim).
        """
        residual = x            # start skip connection
        x = self.norm_in(x)     # norm layer
        x = self.attn(    # multihead self attention
            x, 
            cache_len=cache_len,
            rope=rope,
            attn_mask=attn_mask,
            )
        x = self.atomic_sum(x, residual) # join skip connection
        residual = x                     # start skip connection
        x = self.norm_out(x)             # norm layer
        x = self.mlp(x)                  # mlp block
        x = residual + x                 # join skip connection
        return x
        

class TransformerDecoder(torch.nn.Module):
    def __init__(
            self,
            blocks_config,
            extra_layers_config):
        """
        Generic Transformer Decoder model.
        
        Args:
            blocks_config (list of dict): List of configurations for each TransformerDecoderBlock.
        """
        super().__init__()

        self.depth = len(blocks_config)
        self.blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(block_config) for block_config in blocks_config]
            )
        self.norm = RMSNorm(extra_layers_config["norm"]["embed_dim"])
        head_dim = blocks_config[0]["embed_dim"] // blocks_config[0]["num_heads"]

        self.inv_freq = torch.nn.Parameter(
            precompute_rope_inv_freqs(
                head_dim=head_dim,
                base=1e6,
            ),
            requires_grad=False,
        )

    def forward(
            self,
            x : torch.Tensor,                   # (batch_size, in_seq_len, embed_dim)
            cache_len : int,                    # current stored cache length
    ):
        """
        Forward pass of the Transformer Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_seq_len, embed_dim).
            inv_freq (torch.Tensor): Precomputed inverse frequencies for RoPE of shape (head_dim//2,).
            key_cache list(torch.Tensor): Key cache for attention of size (batch_size, num_heads, max_cache_len, head_dim). Modified in-place.
            value_cache list(torch.Tensor): Value cache for attention of size (batch_size, num_heads, max_cache_len, head_dim). Modified in-place.
            context_len (int): Current context length (position of the next token to generate) used for slicing the caches.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_seq_len, embed_dim).
        """
        # Get dimensions
        _, in_seq_len, _ = x.shape

        # Evaluate RoPE cos and sin once
        # torch._check(cache_len - in_seq_len >= 0, f"Cache length ({cache_len}) must be greater than or equal to input sequence length ({in_seq_len}).") # Ensure cache_len is sufficient for the current input sequence length
        rope = precompute_rope_cos_sin_half(self.inv_freq, in_seq_len, cache_len-in_seq_len) # (1, 1, cache_len+in_seq_len, head_dim//2)

        # Compute attention mask once
        attn_mask =  attention_mask(q_seq_length=in_seq_len, kv_seq_length=cache_len, device=x.device)

        # Decode
        for block in self.blocks:

            # Inference (prefill or decode)
            x = block(
                x,               # (batch_size, in_seq_len, embed_dim)
                cache_len=cache_len, # int
                attn_mask=attn_mask, # (in_seq_len, cache_len)
                rope=rope,       # (1, 1, cache_seq_len, head_dim//2)
            )
            # x is output embeddings: (batch_size, cache_seq_len, embed_dim)

        # Final normalization
        x = self.norm(x) # (batch_size, cache_seq_len, embed_dim)

        return x
    
__all__ = ["TransformerDecoder",]