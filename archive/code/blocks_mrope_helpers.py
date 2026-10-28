"""ARCHIVED: multi-section RoPE (M-RoPE) helpers from blocks/rope.py.

Removed during the 2026-10-28 refactor. They were reachable only from the
``position_embedding_type="mrope"`` branch of TransformerEncoder, which no model
dictionary selected, and the branch was broken (it passed a ``static=`` keyword
that ``precompute_rope_cos_sin_half`` does not accept, and this file passes
``position=`` where the parameter is ``start_pos``). Kept for reference only.
"""
import torch

from transformersurgeon.blocks.positions.rope import precompute_rope_cos_sin_half


def precompute_mrope_inv_freqs(
        head_dim: int,
        section_dims=None,
        base: float = 1e6,
        section_bases=None,
        device="cpu",
        ):
    """
    Precompute inverse frequencies for MRoPE sections.

    Args:
        head_dim: Total head dimension.
        section_dims: Optional list of section dimensions (must sum to head_dim).
        base: Default RoPE base for all sections.
        section_bases: Optional list of per-section bases.
        device: Device where tensors are created.

    Returns:
        Either a single tensor (standard RoPE case) or a list of tensors (MRoPE).
    """
    if section_dims is None:
        return precompute_rope_inv_freqs(head_dim=head_dim, base=base, device=device)

    if sum(section_dims) != head_dim:
        raise ValueError("sum(section_dims) must be equal to head_dim")
    if any(dim <= 0 or dim % 2 != 0 for dim in section_dims):
        raise ValueError("All section dimensions must be positive and even")

    if section_bases is None:
        section_bases = [base] * len(section_dims)
    if len(section_bases) != len(section_dims):
        raise ValueError("section_bases must have the same length as section_dims")

    inv_freq_sections = []
    for dim, sec_base in zip(section_dims, section_bases):
        inv_freq_sections.append(
            1.0 / (sec_base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        )

    return inv_freq_sections

def precompute_mrope_cos_sin_half(
        inv_freq_sections,
        seq_len: int,
        position,
        ):
    """
    Precompute concatenated cos/sin tensors for MRoPE.

    Args:
        inv_freq_sections: List of inverse-frequency tensors (one per section).
        seq_len: Sequence length.
        position: Either a scalar start position (shared across sections) or a list
            with one start position per section.
        static: Same static-position behavior as `precompute_rope_cos_sin_half`.

    Returns:
        Tuple (cos, sin) with concatenated half-rotary dimensions.
    """
    if not isinstance(inv_freq_sections, (list, tuple)) or len(inv_freq_sections) == 0:
        raise ValueError("inv_freq_sections must be a non-empty list or tuple")

    if isinstance(position, (list, tuple)):
        if len(position) != len(inv_freq_sections):
            raise ValueError("position list must match the number of MRoPE sections")
        section_positions = list(position)
    else:
        section_positions = [position] * len(inv_freq_sections)

    cos_sections = []
    sin_sections = []
    for inv_freq, section_pos in zip(inv_freq_sections, section_positions):
        cos_sec, sin_sec = precompute_rope_cos_sin_half(
            inv_freq=inv_freq,
            seq_len=seq_len,
            position=section_pos,
        )
        cos_sections.append(cos_sec)
        sin_sections.append(sin_sec)

    cos = torch.cat(cos_sections, dim=-1)
    sin = torch.cat(sin_sections, dim=-1)
    return cos, sin
