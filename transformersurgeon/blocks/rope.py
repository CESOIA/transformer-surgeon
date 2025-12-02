import torch

def precompute_mrope_inv_freqs(
        section_dims : list = [16, 24, 24],
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
    inv_freqs = []
    for i, dim in enumerate(section_dims):
        base_prime = base ** ((i+1)/len(section_dims))
        inv_freq = 1.0 / (base_prime ** (torch.arange(0, dim, 2, device=device).float() / dim))
        inv_freqs.append(inv_freq)
    inv_freqs = torch.cat(inv_freqs, dim=0) # (sum(section_dim)//2) -> (rotated_dim//2)
    return inv_freqs 

def precompute_mrope_cos_sin_half(
        inv_freq : torch.Tensor,  # (rotated_dim//2,)
        seq_len : int, 
        position : int = None,
        ):
    """
    Precompute the cosine and sine values for RoPE.
    This can be performed once for all the blocks in the model, once per inference.

    Args:
        inv_freq: Precomputed inverse frequencies (from precompute_rope_inv_freqs)
        seq_len: Length of the input sequence
        position: Specific position for decoding (None for encoding/prefill)

    Returns:
        cos: Cosine values for RoPE (half-size)
        sin: Sine values for RoPE (half-size)
    """
    device = inv_freq.device
    # Broadcast cos and sin to match x's shape
    if position is None: # prefill or encode - use all positions from 0 to seq_len-1
        t = torch.arange(seq_len, device=device).float()
        angles = t.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (seq_len, rotated_dim//2)
        angles = angles.unsqueeze(0).unsqueeze(0)         # (1, 1, seq_len, rotated_dim//2)
    else: # decode - use only the specified position
        angles = position * inv_freq # (sections_num, rotated_dim//2)
        angles = angles.view(1, 1, 1, -1)  # (1, 1, 1, rotated_dim//2)

    cos = torch.cos(angles)  # (1, 1, 1, rotated_dim//2)
    sin = torch.sin(angles)  # (1, 1, 1, rotated_dim//2)
    return cos, sin

def apply_mrope_multihead(
        x : torch.Tensor,          # (batch_size, num_heads, seq_len, head_dim)
        cos : torch.Tensor,        # Precomputed half-size cosine values for RoPE (1, 1, seq_len, rotated_dim//2)
        sin : torch.Tensor,        # Precomputed half-size sine values for RoPE (1, 1, seq_len, rotated_dim//2)
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
    _, _, _, half_rotated_dim = cos.shape  # (1, 1, seq_len, rotated_dim//2)
    rotated_dim = half_rotated_dim * 2

    # Generate even and odd indices (safest for export)
    even_idx = torch.arange(0, rotated_dim, 2, device=x.device)
    odd_idx = torch.arange(1, rotated_dim, 2, device=x.device)

    # Slice even dims
    x_even = torch.index_select(x, -1, even_idx)
    x_odd = torch.index_select(x, -1, odd_idx)

    # Apply rotation
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    # Re-interleave even and odd dimensions
    out = torch.zeros_like(x)
    out[..., even_idx] = y_even
    out[..., odd_idx] = y_odd
    out[..., rotated_dim:] = x[..., rotated_dim:] # copy unrotated tail

    return out

__all__ = [
    "precompute_mrope_inv_freqs",
    "precompute_mrope_cos_sin_half",
    "apply_mrope_multihead",
]
