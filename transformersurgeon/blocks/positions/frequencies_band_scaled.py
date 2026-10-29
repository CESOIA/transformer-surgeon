"""Rotary-embedding frequencies with band-wise scaling for long contexts.

Hugging Face configurations name this scaling ``rope_type = "llama3"``; the
catalog registers it under that name so configurations resolve to it directly.
"""

import math

import torch

from .frequencies_default import inv_freq_default


def inv_freq_band_scaled(head_dim: int, params: dict, device="cpu") -> torch.Tensor:
    """Band-wise frequency scaling: low frequencies (wavelength above
    ``original_max_position_embeddings / low_freq_factor``) are divided by
    ``factor``, high frequencies are kept, and the band in between is
    interpolated smoothly. The attention factor is 1."""
    for key in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"):
        if key not in params:
            raise ValueError(f"Band-scaled RoPE needs '{key}' in the config's RoPE parameters.")
    inv_freq = inv_freq_default(head_dim, params, device)
    factor = float(params["factor"])
    low_freq_factor = float(params["low_freq_factor"])
    high_freq_factor = float(params["high_freq_factor"])
    old_context_len = float(params["original_max_position_embeddings"])

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


__all__ = ["inv_freq_band_scaled"]
