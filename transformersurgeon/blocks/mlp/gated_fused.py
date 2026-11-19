"""Gated perceptron with one fused input projection: ``down_proj(act(x1) * x2)``, ``[x1, x2] = up_proj(x)``."""

import torch

from ..layers import linear_from_spec
from .activations import build_activation


class MLPGatedFused(torch.nn.Module):
    """Feed-forward block whose gate and up projections are one linear layer.

    ``up_proj`` produces ``2 * hidden_dim`` channels; the first half is the gate
    (goes through the activation), the second half multiplies it element-wise
    (the SwiGLU layout of DINOv2-giant). Same arguments as
    :class:`~transformersurgeon.blocks.mlp.plain.MLP`; ``hidden_dim`` is the
    width *after* the split.
    """

    projections = ("up_proj", "down_proj")

    def __init__(self, embed_dim, hidden_dim, activation="silu", linear_specs=None, dtype=None):
        super().__init__()
        specs = linear_specs or {}
        self.up_proj = linear_from_spec(specs.get("up_proj"), embed_dim, 2 * hidden_dim, dtype=dtype)
        if self.up_proj.out_features % 2:
            raise ValueError(f"up_proj of a fused gated MLP needs an even width, got {self.up_proj.out_features}.")
        self.activation = build_activation(activation)
        self.down_proj = linear_from_spec(specs.get("down_proj"), self.up_proj.out_features // 2, embed_dim,
                                          dtype=dtype)

    def forward(self, x):
        x_gate, x_up = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.activation(x_gate) * x_up)


__all__ = ["MLPGatedFused"]
