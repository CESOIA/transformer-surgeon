import torch

def precompute_rope_inv_freqs(head_dim=56, base=10000, device="cpu"):
    """
    Precompute the inverse frequencies for Rotary Positional Embeddings (RoPE).
    This can be performed once before deployment.

    Args:
        head_dim: Dimension of each attention head (must be even)
        base: Base value for frequency calculation

    Returns:
        inv_freq: Inverse frequencies for RoPE
    """
    # Calculate the inverse frequencies along the head dimension
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    return inv_freq

def precompute_rope_cos_sin_half(inv_freq, seq_len, position):
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
        angles = torch.outer(t, inv_freq).unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, head_dim)
        cos = torch.cos(angles) 
        sin = torch.sin(angles)
    else: # decode - use only the specified position
        angles = position * inv_freq
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, head_dim)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, head_dim)
    return cos, sin

def apply_rope_multihead(x, cos, sin):
    """
    Apply the rotation to the input tensor x (query or key) using precomputed cosine and sine values.

    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos: Precomputed half-size cosine values for RoPE
        sin: Precomputed half-size sine values for RoPE

    Returns:
        out: Rotated tensor of the same shape as x
    """
    # Slice even dims
    x_even = x[..., 0::2] # (batch_size, num_heads, seq_len, head_dim/2)
    x_odd = x[..., 1::2]  # (batch_size, num_heads, seq_len, head_dim/2)

    # Apply rotation
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    # Re-interleave even and odd dimensions
    out = torch.empty_like(x)
    out[..., 0::2] = y_even
    out[..., 1::2] = y_odd

    return out

__all__ = [
    "precompute_rope_inv_freqs",
    "precompute_rope_cos_sin_half",
    "apply_rope_multihead",
]