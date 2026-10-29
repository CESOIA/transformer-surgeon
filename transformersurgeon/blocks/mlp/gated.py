"""Gated perceptron: ``down_proj(activation(gate_proj(x)) * up_proj(x))``."""

import torch

from ..layers import linear_from_spec
from .activations import build_activation


class MLPGated(torch.nn.Module):
    """Feed-forward block whose hidden layer is gated element-wise.

    Same arguments as :class:`~transformersurgeon.blocks.mlp.plain.MLP`, plus
    ``gate_proj`` in ``linear_specs``.
    """

    projections = ("up_proj", "gate_proj", "down_proj")

    def __init__(self, embed_dim, hidden_dim, activation="gelu", linear_specs=None, dtype=None):
        super().__init__()
        specs = linear_specs or {}
        self.up_proj = linear_from_spec(specs.get("up_proj"), embed_dim, hidden_dim, dtype=dtype)
        self.gate_proj = linear_from_spec(specs.get("gate_proj"), embed_dim, hidden_dim, dtype=dtype)
        self.activation = build_activation(activation)
        self.down_proj = linear_from_spec(specs.get("down_proj"), self.up_proj.out_features, embed_dim, dtype=dtype)

    def forward(self, x):
        x_up = self.up_proj(x)
        x_gate = self.gate_proj(x)
        return self.down_proj(self.activation(x_gate) * x_up)


__all__ = ["MLPGated"]
