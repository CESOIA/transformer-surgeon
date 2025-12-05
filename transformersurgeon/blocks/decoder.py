import torch
from .mha import MHACausal
from .mlp import MLP
from .mlp import MLPGated
from .norm import RMSNorm
from .rope import precompute_rope_cos_sin_half

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

        # Extract configuration (mandatory)
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.mlp_hidden_dim = config["mlp_hidden_dim"]
        self.mlp_activation = config["mlp_activation"]
        self.mha_type = config["mha_type"]
        self.mlp_type = config["mlp_type"]
        self.norm_type = config["norm_type"]
        self.bias_required = config["bias_required"]

        # Extract configuration (optional)
        self.kv_num_heads = None if "kv_num_heads" not in config else config["kv_num_heads"]
        self.compression_config = {
            "attn": None,
            "mlp": None,
        } if "compression_config" not in config else config["compression_config"]

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
                compression_config=self.compression_config["attn"])
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

    def forward(
            self,
            x,
            prefill=True,
            key_cache=None,
            value_cache=None,
            rope=None,):
        """
        Forward pass of the Transformer Decoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            prefill (bool): Whether in prefill mode (True) or decode mode (False). Prefill mode does not require key_cache/value_cache.
            key_cache (torch.Tensor, optional): Key cache for attention of size (batch_size, num_heads, seq_len, head_dim).
            value_cache (torch.Tensor, optional): Value cache for attention of size (batch_size, num_heads, seq_len, head_dim).
            rope (tuple(torch.Tensor, torch.Tensor), optional): Precomputed RoPE cos/sin for queries and keys of size (1, 1, seq_len, head_dim). If None, no RoPE is applied.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            torch.Tensor: Updated key cache of shape (batch_size, num_heads, total_seq_len+1, head_dim).
            torch.Tensor: Updated value cache of shape (batch_size, num_heads, total_seq_len+1, head_dim).
        """
        residual = x            # start skip connection
        x = self.norm_in(x)     # norm layer
        x, k, v = self.attn(    # multihead self attention
            x, 
            key_cache=key_cache,
            value_cache=value_cache,
            rope=rope,
            prefill=prefill
            )
        x = x + residual        # join skip connection
        residual = x            # start skip connection
        x = self.norm_out(x)    # norm layer
        x = self.mlp(x)         # mlp block
        x = residual + x        # join skip connection
        return x, k, v
        

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

        self.blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(block_config) for block_config in blocks_config]
            )
        self.norm = RMSNorm(extra_layers_config["norm"]["embed_dim"])
        self.cache_lengths = self._precompute_cache_lengths()
        self.cache_cumlengths = [0] + torch.cumsum(torch.tensor(self.cache_lengths), dim=0).tolist()

    def forward(
            self,
            x : torch.Tensor,               # (batch_size, seq_len, embed_dim)
            inv_freq : torch.Tensor = None, # (head_dim//2,)
            key_cache = None,               # (batch_size, seq_len, cumsum(cache_lengths))
            value_cache = None,             # (batch_size, seq_len, cumsum(cache_lengths))
            position = None,):
        """
        Forward pass of the Transformer Decoder. Supports:
        - Prefill mode (position is None): processes the entire input sequence and generates key/value caches for the whole sequence.
        - Decode mode (position is provided): processes one token at a time, using the provided key/value caches. It returns the new key/value cache elements for the current position, to be appended externally to the existing caches.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            inv_freq (torch.Tensor, optional): Precomputed inverse frequencies for RoPE of shape (head_dim//2,). If None, no RoPE is applied.
            key_cache (torch.Tensor, optional): Key cache for attention of size (batch_size, seq_len, cumsum(self.cache_lengths)). Can be None if position is None (prefill).
            value_cache (torch.Tensor, optional): Value cache for attention of size (batch_size, seq_len, cumsum(self.cache_lengths)). Can be None if position is None (prefill).
            position (int, optional): Specific position for decoding. If None, prefill mode is used.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            torch.Tensor: key_cache generated after prefill or new key_cache element (batch_size, seq_len, cumsum(self.cache_lengths)).
            torch.Tensor: value_cache generated after prefill or new value_cache element (batch_size, seq_len, cumsum(self.cache_lengths)).
        """
        
        batch_size, seq_len, embed_dim = x.shape 

        # Evaluate RoPE cos and sin once
        if inv_freq is not None:
            rope = precompute_rope_cos_sin_half(inv_freq, seq_len, position) # (1, 1, seq_len, head_dim//2)
        else:
            rope = None
        
        # Decode
        prefill = position is None # if position is not provided, we are in prefill mode
        seq_len = x.size(1)
        cache_seq_len = seq_len if key_cache is None else key_cache.size(1)
        new_key_cache = [] # write-back key cache
        new_value_cache = [] # write-back value cache
        for i, block in enumerate(self.blocks):

            # Number of key and value heads
            kv_num_heads = block.kv_num_heads if block.kv_num_heads is not None else block.num_heads
            # Head dim (evaluated on the number of query heads)
            head_dim = embed_dim // block.num_heads
            # Cache lengths for this block
            cache_len = self.cache_lengths[i]
            cache_start, cache_stop = self.cache_cumlengths[i], self.cache_cumlengths[i+1]

            if not prefill: # Prepare for Decode
                # Slice cache for this block
                k = key_cache[..., cache_start:cache_stop]
                v = value_cache[..., cache_start:cache_stop]
                # Unpack cache
                k = k.view(batch_size, cache_seq_len, kv_num_heads, head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, cache_seq_len, head_dim)
                v = v.view(batch_size, cache_seq_len, kv_num_heads, head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, cache_seq_len, head_dim)
            else: # Prefill
                k = None
                v = None

            # Inference (prefill or decode)
            x, k, v = block(
                x,                       # (batch_size, seq_len, embed_dim)
                key_cache=k,     # (batch_size, num_heads, total_seq_len, head_dim) or None
                value_cache=v, # (batch_size, num_heads, total_seq_len, head_dim) or None
                rope=rope,               # (1, 1, seq_len, head_dim//2)
                prefill=prefill,
            )
            # x     : (batch_size, seq_len, embed_dim)
            # new_k : (batch_size, num_heads, seq_len, head_dim)
            # new_v : (batch_size, num_heads, seq_len, head_dim)

            # Pack new cache
            k = k.permute(0, 2, 1, 3).reshape(batch_size, -1, cache_len)
            v = v.permute(0, 2, 1, 3).reshape(batch_size, -1, cache_len)
            new_key_cache.append(k)
            new_value_cache.append(v)
        
        new_key_cache = torch.cat(new_key_cache, dim=-1) # (batch_size, cache_seq_len, cumsum(cache_lengths))
        new_value_cache = torch.cat(new_value_cache, dim=-1) # (batch_size, cache_seq_len, cumsum(cache_lengths))

        # Final normalization
        x = self.norm(x) # (batch_size, seq_len, embed_dim)

        return x, new_key_cache, new_value_cache
    
    def _precompute_cache_lengths(self):
        """
        Calculate the cache lengths for each layer based on their configuration.

        Returns:
            list : Cache lengths for each layer.
        """
        cache_lengths = []
        for block in self.blocks:
            kv_num_heads = block.kv_num_heads if block.kv_num_heads is not None else block.num_heads
            head_dim = block.embed_dim // block.num_heads
            cache_len = kv_num_heads * head_dim
            cache_lengths.append(cache_len)
        return cache_lengths
    
__all__ = ["TransformerDecoder",]