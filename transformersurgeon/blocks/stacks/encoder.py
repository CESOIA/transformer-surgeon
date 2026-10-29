"""Export-ready bidirectional transformer encoder."""

import torch

from ..catalog import build_block, norm_atom_name
from ..layers import AtomicSum, LayerScale
from .config import CustomEncoderConfigCompress


def _norm(config):
    name = norm_atom_name(config.norm_type, getattr(config, "rmsnorm_impl", "rmsnorm_prescaled"))
    eps = getattr(config, "rms_norm_eps", None) or getattr(config, "layer_norm_eps", None) or 1e-5
    return build_block("norm", name, config.hidden_size, eps=eps)


class TransformerEncoderBlock(torch.nn.Module):
    """One encoder block. ``norm_position`` selects the residual ordering:

    - ``"pre"``:  ``x + attn(norm_in(x))``, then ``x + mlp(norm_out(x))``;
    - ``"post"``: ``norm_in(x + attn(x))``, then ``norm_out(x + mlp(x))``.

    With ``layer_scale`` (pre-norm only) each branch output is scaled per channel
    (``scale_attn``, ``scale_mlp``) before the residual addition.
    """

    def __init__(self, config, block_index):
        super().__init__()
        self.norm_position = getattr(config, "norm_position", "pre")
        if self.norm_position not in ("pre", "post"):
            raise ValueError(f"Unsupported norm_position: {self.norm_position}")
        self.norm_in = _norm(config)
        self.norm_out = _norm(config)
        self.attn = build_block(
            "attention", config.attn_type,
            config.hidden_size, config.num_attention_heads,
            kv_num_heads=getattr(config, "num_key_value_heads", None),
            attn_impl=getattr(config, "attn_impl", None),
            use_sdpa=getattr(config, "use_sdpa", False),
            linear_specs=config.block_layer_specs(block_index, "attn"),
        )
        self.mlp = build_block(
            "mlp", config.mlp_type, config.hidden_size, config.intermediate_size,
            activation=config.hidden_act, linear_specs=config.block_layer_specs(block_index, "mlp"),
        )
        if getattr(config, "layer_scale", False):
            if self.norm_position != "pre":
                raise ValueError("layer_scale requires norm_position='pre'")
            self.scale_attn = LayerScale(config.hidden_size)
            self.scale_mlp = LayerScale(config.hidden_size)
        else:
            self.scale_attn = self.scale_mlp = torch.nn.Identity()
        self.atomic_sum = AtomicSum()

    def forward(self, x, rope=None):
        if self.norm_position == "pre":
            x = self.atomic_sum(self.scale_attn(self.attn(self.norm_in(x), rope=rope)), x)
            return self.atomic_sum(self.scale_mlp(self.mlp(self.norm_out(x))), x)
        x = self.norm_in(self.atomic_sum(self.attn(x, rope=rope), x))
        return self.norm_out(self.atomic_sum(self.mlp(x), x))


class TransformerEncoder(torch.nn.Module):
    """Stack of encoder blocks with an optional final normalisation.

    ``forward(x)`` takes token embeddings ``(seq_len, hidden)``; embeddings and
    heads are applied outside the stack.
    """

    config_class = CustomEncoderConfigCompress

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.depth = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.blocks = torch.nn.ModuleList(
            [TransformerEncoderBlock(config, block_index=i) for i in range(self.depth)])
        self.norm = _norm(config) if getattr(config, "use_final_norm", True) else torch.nn.Identity()

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("Expected token embedding input of shape (seq, hidden)")
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


__all__ = ["TransformerEncoderBlock", "TransformerEncoder"]
