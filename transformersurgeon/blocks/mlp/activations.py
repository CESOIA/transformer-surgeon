"""Activation functions available to the MLP atoms, by the name used in model configurations."""

import torch

ACTIVATIONS = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "silu": torch.nn.SiLU,
}


def build_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation {name!r}; available: {sorted(ACTIVATIONS)}.")
    return ACTIVATIONS[name]()


__all__ = ["ACTIVATIONS", "build_activation"]
