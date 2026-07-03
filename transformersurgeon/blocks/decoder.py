import torch
from .mha import MHACausal
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
        self.cache_impl = getattr(config, "cache_impl", "mutable")
        self.dtype = config.dtype

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
            self.norm_in = RMSNorm(self.embed_dim, dtype=self.dtype)
            self.norm_out = RMSNorm(self.embed_dim, dtype=self.dtype)
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
                max_cache_len=self.max_cache_len,
                cache_impl=self.cache_impl,
                dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported MHA type: {self.mha_type}")

        # Instantiate multi-layer perceptron module
        if self.mlp_type == "mlp_gated":
            self.mlp = MLPGated(
                self.embed_dim,
                self.mlp_hidden_dim,
                bias_required=self.bias_required["mlp"],
                activation=self.mlp_activation,
                compression_config=self.compression_config["mlp"],
                dtype=self.dtype)
        elif self.mlp_type == "mlp":
            self.mlp = MLP(
                self.embed_dim,
                self.mlp_hidden_dim,
                bias_required=self.bias_required["mlp"],
                activation=self.mlp_activation,
                compression_config=self.compression_config["mlp"],
                dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported MLP type: {self.mlp_type}")

        # Instantiate skip connection atomicity module
        self.atomic_sum = AtomicSum()

    def forward(
            self,
            x,
            pos_id,
            pos_id_list,
            mask_penalty,
            key_cache=None,
            value_cache=None,
            rope=None,
    ):
        """
        Forward pass of the Transformer Decoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (in_seq_len, embed_dim).
            pos_id (torch.LongTensor): Current position id, shape (1,).
            pos_id_list (tuple): Precomputed (q_pos, k_pos) index grids for masking.
            mask_penalty (torch.Tensor): Additive penalty applied to masked positions.
            key_cache/value_cache (torch.Tensor, optional): Incoming fixed-size KV
                caches for the ``io_*`` cache implementations (ignored by ``mutable``).
            rope (tuple(torch.Tensor, torch.Tensor), optional): Precomputed RoPE cos/sin tensors.

        Returns:
            mutable mode: output embeddings (seq_len, embed_dim).
            io modes: (output, updated_key_cache, updated_value_cache).
        """
        residual = x            # start skip connection
        x = self.norm_in(x)     # norm layer
        attn_out = self.attn(   # multihead self attention
            x,
            pos_id=pos_id,
            pos_id_list=pos_id_list,
            mask_penalty=mask_penalty,
            key_cache=key_cache,
            value_cache=value_cache,
            rope=rope,
            )
        if self.cache_impl == "mutable":
            x = attn_out
            key_cache_out = value_cache_out = None
        else:
            x, key_cache_out, value_cache_out = attn_out
        x = self.atomic_sum(x, residual) # join skip connection
        residual = x                     # start skip connection
        x = self.norm_out(x)             # norm layer
        x = self.mlp(x)                  # mlp block
        x = self.atomic_sum(x, residual) # join skip connection
        if self.cache_impl == "mutable":
            return x
        return x, key_cache_out, value_cache_out
        

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
        self.dtype = config.dtype
        self.blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(config, block_index=i) for i in range(self.depth)]
            )
        self.norm = RMSNorm(config.hidden_size, self.dtype)
        head_dim = config.hidden_size // config.num_attention_heads

        self.max_cache_len = config.max_cache_len
        self.cache_impl = getattr(config, "cache_impl", "mutable")

        # Precompute position ids useful for attention mask generation
        q_pos = torch.arange(self.max_cache_len, dtype=torch.long)[:, None] # (max_cache_len, 1)
        k_pos = torch.arange(self.max_cache_len, dtype=torch.long)[None, :] # (1, max_cache_len)
        self.register_buffer("q_pos", q_pos, persistent=True)
        self.register_buffer("k_pos", k_pos, persistent=True)

        # Precompute RoPE once for all position in kv_cache (up to max_cache_len)
        inv_freq = precompute_rope_inv_freqs(
            head_dim=head_dim,
            base=1e6,
        )
        rope = precompute_rope_cos_sin_half(
            inv_freq,
            self.max_cache_len,
            start_pos=0,
        )
        self.register_buffer("rope_cos", rope[0].to(config.dtype), persistent=True)
        self.register_buffer("rope_sin", rope[1].to(config.dtype), persistent=True)

        # Store the penalty value as a proper 1-element tensor buffer
        self.register_buffer('mask_penalty', torch.tensor(-10000.0, dtype=config.dtype), persistent=True)


    def forward(
            self,
            x : torch.Tensor,  # (in_seq_len, embed_dim)
            pos_id : torch.LongTensor,  # (1,)
            key_caches=None,   # io modes: List[Tensor], one per block
            value_caches=None,
    ):
        """
        Forward pass of the Transformer Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (in_seq_len, embed_dim).
            pos_id (torch.LongTensor): Current position id tensor of shape (1,).
            key_caches/value_caches (List[torch.Tensor], optional): Per-block
                incoming KV caches for the ``io_*`` implementations. If omitted,
                each block falls back to its internal buffer.

        Returns:
            mutable mode: output tensor (in_seq_len, embed_dim).
            io modes: (output, new_key_caches, new_value_caches) where the caches
                are per-block Lists of length ``depth``.
        """
        # Decode
        if self.cache_impl == "mutable":
            for block in self.blocks:
                x = block(
                    x,               # (in_seq_len, embed_dim)
                    pos_id=pos_id,   # (1,)
                    pos_id_list=(self.q_pos, self.k_pos),
                    mask_penalty=self.mask_penalty,
                    rope=(self.rope_cos, self.rope_sin),
                )
            x = self.norm(x)
            return x

        # io modes: thread per-block caches in and out.
        new_key_caches, new_value_caches = [], []
        for i, block in enumerate(self.blocks):
            kc = key_caches[i] if key_caches is not None else None
            vc = value_caches[i] if value_caches is not None else None
            x, kc_out, vc_out = block(
                x,
                pos_id=pos_id,
                pos_id_list=(self.q_pos, self.k_pos),
                mask_penalty=self.mask_penalty,
                key_cache=kc,
                value_cache=vc,
                rope=(self.rope_cos, self.rope_sin),
            )
            new_key_caches.append(kc_out)
            new_value_caches.append(vc_out)

        x = self.norm(x) # (1, embed_dim)
        return x, new_key_caches, new_value_caches
    
__all__ = ["TransformerDecoder",]