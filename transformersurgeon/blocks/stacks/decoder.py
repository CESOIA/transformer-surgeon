"""Export-ready causal transformer decoder."""

import torch

from ..catalog import build_block, norm_atom_name
from ..layers import AtomicSum
from .config import CustomDecoderConfigCompress
from ..attention.kernels.paged import identity_page_table, num_kv_pages
from ..attention.kv_cache import KV_CACHES
from ..positions.rope import (
    rope_parameters_from_config,
    precompute_rope_inv_freqs_for,
    precompute_rope_cos_sin_half,
)


def _norm(config, dtype):
    name = norm_atom_name(config.norm_type, getattr(config, "rmsnorm_impl", "rmsnorm_prescaled"))
    return build_block("norm", name, config.hidden_size, dtype=dtype, eps=getattr(config, "rms_norm_eps", 1e-5))


class TransformerDecoderBlock(torch.nn.Module):
    """One decoder block: ``x + attn(norm_in(x))`` then ``x + mlp(norm_out(x))``.

    The attention, MLP and normalisation atoms are taken from the block catalog
    by the names in ``config`` (``attn_type``, ``mlp_type``, ``norm_type`` +
    ``rmsnorm_impl``). Submodule names (``norm_in``, ``attn``, ``norm_out``,
    ``mlp``) are a fixed convention shared with the converter.
    """

    def __init__(self, config, block_index):
        super().__init__()
        self.cache_impl = getattr(config, "cache_impl", "mutable")
        self.returns_cache = KV_CACHES[self.cache_impl].graph_io
        dtype = config.dtype
        self.norm_in = _norm(config, dtype)
        self.norm_out = _norm(config, dtype)
        self.attn = build_block(
            "attention", config.attn_type,
            config.hidden_size, config.num_attention_heads,
            kv_num_heads=getattr(config, "num_key_value_heads", None),
            attn_impl=getattr(config, "attn_impl", None),
            use_sdpa=getattr(config, "use_sdpa", False),
            linear_specs=config.block_layer_specs(block_index, "attn"),
            dtype=dtype,
            cache_impl=self.cache_impl,
            max_cache_len=config.max_cache_len,
            add_batch_dim=getattr(config, "add_batch_dim", False),
            masked_softmax=getattr(config, "masked_softmax", False),
        )
        self.mlp = build_block(
            "mlp", config.mlp_type, config.hidden_size, config.intermediate_size,
            activation=config.hidden_act, linear_specs=config.block_layer_specs(block_index, "mlp"), dtype=dtype,
        )
        self.atomic_sum = AtomicSum()

    def forward(self, x, pos_id, attn_mask, key_cache=None, value_cache=None, rope=None, paged=None):
        """Returns ``x`` for an internal cache, else ``(x, key_cache, value_cache)``."""
        residual = x
        attn_out = self.attn(self.norm_in(x), pos_id=pos_id, attn_mask=attn_mask,
                             key_cache=key_cache, value_cache=value_cache, rope=rope, paged=paged)
        if self.returns_cache:
            x, key_cache_out, value_cache_out = attn_out
        else:
            x = attn_out
        x = self.atomic_sum(x, residual)
        residual = x
        x = self.atomic_sum(self.mlp(self.norm_out(x)), residual)
        if self.returns_cache:
            return x, key_cache_out, value_cache_out
        return x


class TransformerDecoder(torch.nn.Module):
    """Stack of decoder blocks with rotary positions, causal masking and a final normalisation.

    Embedding and output head are not part of the stack: exporters wrap them
    around it. ``forward(x, pos_id, key_caches=None, value_caches=None)`` takes
    ``x`` of shape ``(q_len, hidden)`` (``(1, q_len, hidden)`` with
    ``add_batch_dim``) and the position of its first token. With an internal
    cache it returns the hidden states; with graph-I/O caches it returns
    ``(hidden, new_key_caches, new_value_caches)`` (one entry per block).
    """

    config_class = CustomDecoderConfigCompress

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.depth = config.num_hidden_layers
        self.dtype = config.dtype
        self.blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(config, block_index=i) for i in range(self.depth)])
        self.norm = _norm(config, self.dtype)

        self.max_cache_len = config.max_cache_len
        self.cache_impl = getattr(config, "cache_impl", "mutable")
        self.kv_cache = KV_CACHES[self.cache_impl]
        self.add_batch_dim = getattr(config, "add_batch_dim", False)

        self.register_buffer("q_pos", torch.arange(self.max_cache_len, dtype=torch.long)[:, None], persistent=True)
        self.register_buffer("k_pos", torch.arange(self.max_cache_len, dtype=torch.long)[None, :], persistent=True)

        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = precompute_rope_inv_freqs_for(
            getattr(config, "rope_type", "default"), head_dim, rope_parameters_from_config(config))
        cos, sin = precompute_rope_cos_sin_half(inv_freq, self.max_cache_len, start_pos=0)
        self.register_buffer("rope_cos", cos.to(config.dtype), persistent=True)
        self.register_buffer("rope_sin", sin.to(config.dtype), persistent=True)
        if self.cache_impl == "io_paged":
            self.register_buffer("rope_cos_sin", torch.cat([cos, sin], dim=-1).squeeze(1).unsqueeze(0).float(),
                                 persistent=False)
            self.register_buffer("kv_page_table", identity_page_table(num_kv_pages(self.max_cache_len)),
                                 persistent=False)

        # Finite large negative penalty (float16 minimum) rather than -inf: some
        # models produce attention logits far above 1e4, so a smaller penalty
        # would leak probability onto masked positions.
        self.register_buffer("mask_penalty", torch.tensor(-65504.0, dtype=config.dtype), persistent=True)

    def _forward_paged(self, x, pos_id, key_caches):
        in_seq_len = x.shape[1] if self.add_batch_dim else x.shape[0]
        start = pos_id.to(torch.int32)
        paged = (start + in_seq_len, self.rope_cos_sin, start, self.kv_page_table)
        new_pools = []
        for i, block in enumerate(self.blocks):
            x, pool, _ = block(x, pos_id=pos_id, attn_mask=None, key_cache=key_caches[i], paged=paged)
            new_pools.append(pool)
        return self.norm(x), new_pools, None

    def forward(self, x, pos_id, key_caches=None, value_caches=None):
        if self.cache_impl == "io_paged":
            return self._forward_paged(x, pos_id, key_caches)

        in_seq_len = x.shape[1] if self.add_batch_dim else x.shape[0]
        attn_mask, positions = self.kv_cache.decoder_mask(self, pos_id, in_seq_len)
        rope_pos = (self.rope_cos[positions], self.rope_sin[positions])

        if not self.kv_cache.graph_io:
            for block in self.blocks:
                x = block(x, pos_id=pos_id, attn_mask=attn_mask, rope=rope_pos)
            return self.norm(x)

        new_key_caches, new_value_caches = [], []
        for i, block in enumerate(self.blocks):
            kc = key_caches[i] if key_caches is not None else None
            vc = value_caches[i] if value_caches is not None else None
            x, kc_out, vc_out = block(x, pos_id=pos_id, attn_mask=attn_mask, key_cache=kc, value_cache=vc,
                                      rope=rope_pos)
            new_key_caches.append(kc_out)
            new_value_caches.append(vc_out)
        return self.norm(x), new_key_caches, new_value_caches


__all__ = ["TransformerDecoderBlock", "TransformerDecoder"]
