import torch
from . import MHACausal
from . import MLP
from . import MLPGated
from . import RMSnorm
from . import precompute_rope_cos_sin_half

class TransformerDecoderBlock(torch.nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_hidden_dim,
            gated_mlp=False,
            q_num_heads=None):
        super().__init__()
        """
        Generic Transformer Decoder Block.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads (or KV heads if q_num_heads is given).
            mlp_hidden_dim (int): Hidden dimension of the MLP.
            gated_mlp (bool): Whether to use gated MLP.
            q_num_heads (int, optional): Number of Q heads (enables GQA). If None, set to num_heads.
        """

        self.num_heads = num_heads

        self.norm_in = RMSnorm(embed_dim)
        self.attn = MHACausal(embed_dim, num_heads)
        self.norm_out = RMSnorm(embed_dim)
        if gated_mlp:
            self.mlp = MLPGated(embed_dim, mlp_hidden_dim)
        else:
            self.mlp = MLP(embed_dim, mlp_hidden_dim)

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
        x, k, v = self.attn(          # multihead self attention
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
            blocks_config,):
        """
        Generic Transformer Decoder model.
        
        Args:
            blocks_config (list of dict): List of configurations for each TransformerDecoderBlock.
        """
        super().__init__()

        self.blocks = torch.nn.ModuleList([TransformerDecoderBlock(**blocks_config[i]) for i in range(len(blocks_config))])

    def forward(
            self,
            x : torch.Tensor,               # (batch_size, seq_len, embed_dim)
            inv_freq : torch.Tensor = None, # (head_dim//2,)
            key_cache = None,               # (batch_size, seq_len, cumsum(cache_lengths))
            value_cache = None,             # (batch_size, seq_len, cumsum(cache_lengths))
            cache_lengths = None,           # (num_layers,)
            position = None,):
        """
        Forward pass of the Transformer Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            inv_freq (torch.Tensor, optional): Precomputed inverse frequencies for RoPE of shape (head_dim//2,). If None, no RoPE is applied.
            key_cache (torch.Tensor, optional): Key cache for attention of size (batch_size, seq_len, cumsum(cache_lengths)). Can be None if prefill is True.
            value_cache (torch.Tensor, optional): Value cache for attention of size (batch_size, seq_len, cumsum(cache_lengths)). Can be None if prefill is True.
            cache_lengths (torch.Tensor, optional): Lengths of each layer's cache of shape (num_layers,).
            position (int, optional): Specific position for decoding. If None, prefill mode is used.
        """
        
        batch_size, seq_len, embed_dim = x.shape 
        device = x.device

        # Evaluate RoPE cos and sin once
        if inv_freq is not None:
            rope = precompute_rope_cos_sin_half(inv_freq, seq_len, position) # (1, 1, seq_len, head_dim//2)
        else:
            rope = None
        
        # Decode
        prefill = position is None
        cache_cumlen = 0
        wb_key_cache = []
        wb_value_cache = []
        wb_cache_lengths = []
        for i, block in enumerate(self.blocks):

            num_heads = block.num_heads
            head_dim = embed_dim // num_heads

            if position is not None: # Decode
                # Get cache slice
                cache_len = int(cache_lengths[i].item())
                k = key_cache[:, :, cache_cumlen:cache_cumlen+cache_len] # (batch_size, seq_len, length)
                v = value_cache[:, :, cache_cumlen:cache_cumlen+cache_len] # (batch_size, seq_len, length)
                cache_cumlen += cache_len
                # Unpack cache
                total_seq_len = k.size(1)
                k = k.reshape(batch_size, total_seq_len, num_heads, head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, total_seq_len, head_dim)
                v = v.reshape(batch_size, total_seq_len, num_heads, head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, total_seq_len, head_dim)
            else: # Prefill
                k = None
                v = None

            # Prefill/Decode
            x, k, v = block(
                x,                  # (batch_size, seq_len, embed_dim)
                key_cache=k,        # (batch_size, num_heads, total_seq_len, head_dim)
                value_cache=v,      # (batch_size, num_heads, total_seq_len, head_dim)
                rope=rope,          # (1, 1, seq_len, head_dim//2)
                prefill=prefill,
            )

            # Pack cache
            new_total_seq_len = k.size(2) # total_seq_len + 1
            k = k.permute(0, 2, 1, 3).reshape(batch_size, new_total_seq_len, cache_len)
            v = v.permute(0, 2, 1, 3).reshape(batch_size, new_total_seq_len, cache_len)
            wb_cache_lengths.append(k.size(1))
            wb_key_cache.append(k)
            wb_value_cache.append(v)
        
        wb_cache_lengths = torch.tensor(wb_cache_lengths, dtype=torch.long, dtype=device) # (num_layers,)
        wb_key_cache = torch.cat(wb_key_cache, dim=-1) # (batch_size, total_seq_len+1, cumsum(cache_lengths))
        wb_value_cache = torch.cat(wb_value_cache, dim=-1) # (batch_size, total_seq_len+1, cumsum(cache_lengths))

        return x, wb_key_cache, wb_value_cache, wb_cache_lengths
