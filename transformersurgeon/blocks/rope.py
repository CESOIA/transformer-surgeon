import torch


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

def precompute_rope_inv_freqs(
        head_dim : int = 128,
        base : float = 1e6,
        device = "cpu",
        ):
    """
    Precompute the inverse frequencies for Multi-Range Rotary Positional Embeddings (MRoPE).
    This can be performed once before deployment.

    Args:
        section_dims: List of head dimension sizes for each section
        base: Base value for frequency calculation
    
    Returns:
        inv_freqs: Tensor of inverse frequencies for each section
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)) # (head_dim//2,)
    return inv_freq

def precompute_rope_cos_sin_half(
        inv_freq : torch.Tensor,  # (rotated_dim//2,)
        seq_len : int, 
        position : int,
        static : bool = False,
        ):
    """
    Precompute the cosine and sine values for RoPE.
    This can be performed once for all the blocks in the model, once per inference.

    Args:
        inv_freq: Precomputed inverse frequencies (from precompute_rope_inv_freqs)
        seq_len: Length of the input sequence
        position: Specific starting position for the rope sequence

    Returns:
        cos: Cosine values for RoPE (half-size)
        sin: Sine values for RoPE (half-size)
    """
    device = inv_freq.device
    dtype = inv_freq.dtype
    # position = torch.as_tensor(position, device=device, dtype=dtype)
    
    # Broadcast cos and sin to match x's shape
    if static:
        position = torch.as_tensor(position, device=device, dtype=dtype)
        t = torch.arange(seq_len, device=device, dtype=dtype) + position
    else:
        t = torch.arange(position, position + seq_len, device=device, dtype=dtype)
    angles = t.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (seq_len, rotated_dim//2)
    angles = angles.unsqueeze(1).unsqueeze(0)         # (1, seq_len, 1, rotated_dim//2)

    cos = torch.cos(angles)  # (1, seq_len, 1, rotated_dim//2)
    sin = torch.sin(angles)  # (1, seq_len, 1, rotated_dim//2)
    return cos, sin


def precompute_mrope_cos_sin_half(
        inv_freq_sections,
        seq_len: int,
        position,
        static: bool = False,
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
            static=static,
        )
        cos_sections.append(cos_sec)
        sin_sections.append(sin_sec)

    cos = torch.cat(cos_sections, dim=-1)
    sin = torch.cat(sin_sections, dim=-1)
    return cos, sin

def apply_rope_multihead(
        x : torch.Tensor,          # (batch_size, num_heads, seq_len, head_dim)
        cos : torch.Tensor,        # Precomputed half-size cosine values for RoPE (1, seq_len, 1, rotated_dim//2)
        sin : torch.Tensor,        # Precomputed half-size sine values for RoPE (1, seq_len, 1, rotated_dim//2)
        ):
    """
    Apply the rotation to the input tensor x (query or key) using precomputed cosine and sine values.

    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos: Precomputed half-size cosine values for RoPE
        sin: Precomputed half-size sine values for RoPE
        rotated_dim: Size of the portion of the head_dim to be rotated (sum of section_dims)

    Returns:
        out: Rotated tensor of the same shape as x
    """
    # Convert to float32 for numerical stability during rotation
    dtype = x.dtype
    x = x.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    # Split in 2 chunks for the "real" and "imaginary" parts
    x_real, x_imag = torch.chunk(x, 2, dim=-1)  # Each of shape (batch_size, num_heads, seq_len, head_dim//2)

    # Apply rotation
    y_real = x_real * cos - x_imag * sin
    y_imag = x_real * sin + x_imag * cos

    # Re-interleave even and odd dimensions
    out = torch.cat((y_real, y_imag), dim=-1)  # (batch_size, num_heads, seq_len, head_dim)

    # Restore original dtype
    out = out.to(dtype)
    
    return out

__all__ = [
    "precompute_mrope_inv_freqs",
    "precompute_mrope_cos_sin_half",
    "precompute_rope_inv_freqs",
    "precompute_rope_cos_sin_half",
    "apply_rope_multihead",
]
