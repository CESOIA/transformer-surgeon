import torch

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
    # Broadcast cos and sin to match x's shape
    t = torch.arange(position, position+seq_len, device=device).float()
    angles = t.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (seq_len, rotated_dim//2)
    angles = angles.unsqueeze(0).unsqueeze(0)         # (1, 1, seq_len, rotated_dim//2)

    cos = torch.cos(angles)  # (1, 1, 1, rotated_dim//2)
    sin = torch.sin(angles)  # (1, 1, 1, rotated_dim//2)
    return cos, sin

def apply_rope_multihead(
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
    "precompute_rope_inv_freqs",
    "precompute_rope_cos_sin_half",
    "apply_rope_multihead",
]
