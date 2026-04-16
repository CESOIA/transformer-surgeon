import torch
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

    def __init__(self, config, block_index):
        super().__init__()
        """
        Generic Transformer Decoder Block.
        
        Args:
            config: HF-style decoder config with model dimensions and compression metadata.
            block_index (int): Decoder block index used to extract per-layer block metadata.
        """

        # Extract configuration (strict schema).
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.mlp_hidden_dim = config.intermediate_size
        self.mlp_activation = config.hidden_act
        self.mha_type = config.attn_type
        self.mlp_type = config.mlp_type
        self.norm_type = config.norm_type
        self.use_sdpa = config.use_sdpa
        self.max_cache_len = config.max_cache_len

        # Extract configuration (optional)
        self.kv_num_heads = getattr(config, "num_key_value_heads", None)
        self.compression_config = {
            "attn": {},
            "mlp": {},
        }
        prefix = f"blocks.{block_index}."
        for full_path, value in getattr(config, "compression_config", {}).items():
            if not full_path.startswith(prefix):
                continue
            local_path = full_path[len(prefix):]
            if local_path.startswith("attn."):
                self.compression_config["attn"][local_path.split(".", 1)[1]] = value
            elif local_path.startswith("mlp."):
                self.compression_config["mlp"][local_path.split(".", 1)[1]] = value

        self.bias_required = getattr(config, "bias_required", {"attn": {}, "mlp": {}})

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
            last_pos=None,
            attn_mask=None,
            rope=None,
            static=False,
    ):
        """
        Forward pass of the Transformer Decoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_seq_len, embed_dim).
            cache_len (int, optional): Effective sequence length seen by attention.
            attn_mask (torch.Tensor, optional): Attention mask for causal decoding.
            rope (tuple(torch.Tensor, torch.Tensor), optional): Precomputed RoPE cos/sin tensors.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            torch.Tensor: Updated key cache of shape (batch_size, num_heads, total_seq_len+1, head_dim).
            torch.Tensor: Updated value cache of shape (batch_size, num_heads, total_seq_len+1, head_dim).
        """
        residual = x            # start skip connection
        x = self.norm_in(x)     # norm layer
        x = self.attn(    # multihead self attention
            x, 
            last_pos=last_pos,
            rope=rope,
            attn_mask=attn_mask,
            static=static,
            )
        x = self.atomic_sum(x, residual) # join skip connection
        residual = x                     # start skip connection
        x = self.norm_out(x)             # norm layer
        x = self.mlp(x)                  # mlp block
        x = self.atomic_sum(x, residual) # join skip connection
        return x
        

class TransformerDecoder(torch.nn.Module):
    def __init__(self, config):
        """
        Generic Transformer Decoder model.
        
        Args:
            config: HF-style config for converted decoder.
        """
        super().__init__()
        self.config = config

        self.depth = config.num_hidden_layers
        self.blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(config, block_index=i) for i in range(self.depth)]
            )
        self.norm = RMSNorm(config.hidden_size)
        head_dim = config.hidden_size // config.num_attention_heads

        self.inv_freq = torch.nn.Parameter(
            precompute_rope_inv_freqs(
                head_dim=head_dim,
                base=1e6,
            ),
            requires_grad=False,
        )

        self.max_cache_len = config.max_cache_len


    def forward(
            self,
            x : torch.Tensor,                   # (batch_size, in_seq_len, embed_dim)
            last_pos : int,
            static : bool = False,
    ):
        """
        Forward pass of the Transformer Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_seq_len, embed_dim).
            last_pos (int): The position index of the last token in the expected output sequence.
            static (bool): Whether to use static positional indexing (for non-autoregressive decoding).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_seq_len, embed_dim).
        """
        # Get dimensions
        _, in_seq_len, _ = x.shape

        # Evaluate RoPE cos and sin once
        # torch._check(cache_len - in_seq_len >= 0, f"Cache length ({cache_len}) must be greater than or equal to input sequence length ({in_seq_len}).") # Ensure cache_len is sufficient for the current input sequence length
        rope_pos = last_pos - in_seq_len
        rope = precompute_rope_cos_sin_half(
            self.inv_freq,
            in_seq_len,
            rope_pos,
            static=static) # (1, 1, cache_len+in_seq_len, head_dim//2)

        # Compute attention mask once
        attn_mask = attention_mask(
            q_seq_length=in_seq_len,
            kv_seq_length=last_pos,  # effective length
            kv_alloc_length=self.max_cache_len if static else None,
            device=x.device,
        )

        # Decode
        for block in self.blocks:

            # Inference (prefill or decode)
            x = block(
                x,               # (batch_size, in_seq_len, embed_dim)
                last_pos=last_pos, # int
                attn_mask=attn_mask, # (in_seq_len, cache_len)
                rope=rope,       # (1, 1, cache_seq_len, head_dim//2)
                static=static,
            )
            # x is output embeddings: (batch_size, cache_seq_len, embed_dim)

        # Final normalization
        x = self.norm(x) # (batch_size, cache_seq_len, embed_dim)

        return x
    
__all__ = ["TransformerDecoder",]