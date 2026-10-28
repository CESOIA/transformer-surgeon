"""Two-layer perceptron: ``down_proj(activation(up_proj(x)))``."""

import torch

from ..layers import linear_from_spec
from .activations import build_activation


class MLP(torch.nn.Module):
    """Feed-forward block with one hidden layer.

    Args:
        embed_dim: Input and output width.
        hidden_dim: Hidden width.
        activation: Activation name (see ``mlp/activations.py``).
        linear_specs: Optional ``{layer name: spec}`` overriding the geometry of
            ``up_proj``/``down_proj`` (see ``layers.linear_from_spec``).
        dtype: Parameter dtype.
    """

    projections = ("up_proj", "down_proj")

    def __init__(self, embed_dim, hidden_dim, activation="gelu", linear_specs=None, dtype=None):
        super().__init__()
        specs = linear_specs or {}
        self.up_proj = linear_from_spec(specs.get("up_proj"), embed_dim, hidden_dim, dtype=dtype)
        self.activation = build_activation(activation)
        self.down_proj = linear_from_spec(specs.get("down_proj"), self.up_proj.out_features, embed_dim, dtype=dtype)

    def forward(self, x):
        return self.down_proj(self.activation(self.up_proj(x)))


__all__ = ["MLP"]
