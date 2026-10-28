"""Rotary position embedding (RoPE): configuration parsing, tables and application."""

import warnings

import torch

from ..catalog import BLOCK_CATALOG, resolve_atom
from .frequencies_default import precompute_rope_inv_freqs
from .rope_pruning import build_rope_prune_projection


def rope_parameters_from_config(config) -> dict:
    """RoPE parameters of an HF-style config as one flat dict.

    transformers 5 keeps them in ``config.rope_parameters`` (``rope_type``,
    ``rope_theta`` and the scaling fields); 4.x has ``config.rope_scaling``
    (``rope_type`` or legacy ``type``) plus a top-level ``config.rope_theta``.
    Missing ``rope_type`` means plain RoPE (``"default"``).
    """
    params = dict(getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {})
    if "rope_type" not in params and "type" in params:
        params["rope_type"] = params.pop("type")
    params.setdefault("rope_type", "default")
    if params["rope_type"] == "mrope":  # sectioned RoPE marker of some configurations: plain frequencies (as in HF)
        params["rope_type"] = "default"
    if params.get("rope_theta") is None and getattr(config, "rope_theta", None) is not None:
        params["rope_theta"] = config.rope_theta
    return params


def rope_theta_from_config(config, default: float = 1e6) -> float:
    """RoPE base of an HF-style config.

    ``default`` (the historical tsurgeon value) only applies when the config
    has no ``rope_theta``. The encoder path only implements plain RoPE, so a
    scaled variant is reported there; decoders resolve the variant with
    :func:`resolve_rope_type`.
    """
    params = rope_parameters_from_config(config)
    if params["rope_type"] != "default":
        warnings.warn(f"RoPE scaling '{params['rope_type']}' is not implemented here; using plain RoPE.",
                      stacklevel=2)
    theta = params.get("rope_theta")
    return float(default if theta is None else theta)


# rope_type -> inverse-frequency function (head_dim, params, device), from the
# "rope_frequencies" family of the block catalog. Every entry keeps the attention
# scaling at 1 (cos/sin are not rescaled by the decoder). Keys are the
# ``rope_type`` names used by Hugging Face configurations.
ROPE_FREQUENCIES = {name: resolve_atom("rope_frequencies", name) for name in BLOCK_CATALOG["rope_frequencies"]}


def resolve_rope_type(config, indexing_rope_type: str = "default") -> str:
    """RoPE variant a converted decoder must use.

    ``indexing_rope_type`` comes from the model dictionary: a fixed type (for
    example ``"default"``) or ``"auto"`` to take the configuration's own
    ``rope_type``. A fixed type that disagrees with a scaled configuration, or a
    type without an implementation, raises instead of silently using plain RoPE.
    """
    config_type = rope_parameters_from_config(config)["rope_type"]
    if indexing_rope_type == "auto":
        rope_type = config_type
    else:
        rope_type = indexing_rope_type
        if config_type not in ("default", rope_type):
            raise NotImplementedError(
                f"The config asks for RoPE '{config_type}' but the model's indexing fixes "
                f"'{rope_type}'; set the indexing 'rope_type' to 'auto' or '{config_type}'."
            )
    if rope_type not in ROPE_FREQUENCIES:
        raise NotImplementedError(
            f"RoPE type '{rope_type}' is not implemented (available: {sorted(ROPE_FREQUENCIES)})."
        )
    return rope_type


def precompute_rope_inv_freqs_for(rope_type: str, head_dim: int, params: dict, device="cpu") -> torch.Tensor:
    """Inverse frequencies ``(head_dim//2,)`` of one registered RoPE variant."""
    if rope_type not in ROPE_FREQUENCIES:
        raise NotImplementedError(
            f"RoPE type '{rope_type}' is not implemented (available: {sorted(ROPE_FREQUENCIES)})."
        )
    return ROPE_FREQUENCIES[rope_type](head_dim, params, device)


def precompute_rope_cos_sin_half(
        inv_freq : torch.Tensor,  # (rotated_dim//2,)
        seq_len : torch.LongTensor, # (1,)
        start_pos : torch.LongTensor, # (1,)
        ):
    """
    Precompute the cosine and sine values for RoPE.
    This can be performed once for all the blocks in the model, once per inference.

    Args:
        inv_freq: Precomputed inverse frequencies (from precompute_rope_inv_freqs)
        seq_len: RoPE sequence length
        start_pos: Specific starting position for the rope sequence

    Returns:
        cos: Cosine values for RoPE (half-size)
        sin: Sine values for RoPE (half-size)
    """
    device = inv_freq.device
    dtype = inv_freq.dtype
    # position = torch.as_tensor(position, device=device, dtype=dtype)
    
    # Broadcast cos and sin to match x's shape
    t = torch.arange(seq_len, device=device, dtype=dtype) + start_pos

    angles = t.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (seq_len, rotated_dim//2)
    angles = angles.unsqueeze(1)         # (seq_len, 1, rotated_dim//2)

    cos = torch.cos(angles)  # (seq_len, 1, rotated_dim//2)
    sin = torch.sin(angles)  # (seq_len, 1, rotated_dim//2)
    return cos, sin


def apply_rope_multihead(
        x : torch.Tensor,          # (seq_len, num_heads, head_dim)
        cos : torch.Tensor,        # Precomputed half-size cosine values for RoPE (seq_len, 1, rotated_dim//2)
        sin : torch.Tensor,        # Precomputed half-size sine values for RoPE (seq_len, 1, rotated_dim//2)
        ):
    """
    Apply the rotation to the input tensor x (query or key) using precomputed cosine and sine values.

    Args:
        x: Input tensor of shape (seq_len, num_heads, head_dim)
        cos: Precomputed half-size cosine values for RoPE
        sin: Precomputed half-size sine values for RoPE
        rotated_dim: Size of the portion of the head_dim to be rotated (sum of section_dims)

    Returns:
        out: Rotated tensor of the same shape as x
    """
    # Convert to float32 for numerical stability during rotation
    dtype = x.dtype

    # Split in 2 chunks for the "real" and "imaginary" parts
    x_real, x_imag = torch.chunk(x, 2, dim=-1)  # Each of shape (seq_len, num_heads, head_dim//2)

    # Apply rotation
    y_real = x_real * cos - x_imag * sin
    y_imag = x_real * sin + x_imag * cos

    # Re-interleave even and odd dimensions
    out = torch.cat((y_real, y_imag), dim=-1)  # (num_heads, seq_len, head_dim)

    # Restore original dtype
    out = out.to(dtype)
    
    return out


__all__ = [
    "ROPE_FREQUENCIES",
    "rope_parameters_from_config",
    "rope_theta_from_config",
    "resolve_rope_type",
    "precompute_rope_inv_freqs_for",
    "precompute_rope_inv_freqs",
    "precompute_rope_cos_sin_half",
    "apply_rope_multihead",
    "build_rope_prune_projection",
]
