"""Plain rotary-embedding frequencies (no scaling)."""

import torch


def precompute_rope_inv_freqs(
        head_dim : int = 128,
        base : float = 1e6,
        device = "cpu",
        ):
    """Inverse frequencies ``1 / base^(2i / head_dim)``, shape ``(head_dim // 2,)``."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)) # (head_dim//2,)
    return inv_freq


def inv_freq_default(head_dim: int, params: dict, device="cpu") -> torch.Tensor:
    """Plain frequencies with base ``params["rope_theta"]`` (default ``1e6``)."""
    return precompute_rope_inv_freqs(head_dim=head_dim, base=float(params.get("rope_theta", 1e6)), device=device)


__all__ = ["precompute_rope_inv_freqs", "inv_freq_default"]
