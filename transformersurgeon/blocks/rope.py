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

def _derive_rope_prune_pattern(
        keep_mask: torch.Tensor,
        head_dim: int,
        num_heads: int,
        source: str = "",
        ):
    """Extract the length-``head_dim//2`` rotary keep-pattern from a
    structured-pruning keep mask over a q/k projection's full output rows.

    Prunable RoPE requires:
      * the same keep pattern in every head (so one projection works for all
        heads at once);
      * each rotary frequency's "real" and "imaginary" channels (indices ``i``
        and ``i + head_dim//2`` within a head, see ``apply_rope_multihead``)
        kept or pruned together, since they share one ``cos``/``sin`` value.

    Both are satisfied for free by ``structured_pruning`` with
    ``granularity=head_dim // 2`` and ``repeated_pattern=True``: scores are
    reduced in contiguous chunks of size ``head_dim // 2`` (real chunk, then
    imaginary chunk, per head -- see ``reduce_pattern_scores``), and the
    resulting length-``head_dim // 2`` pattern is tiled back over *every*
    chunk, so frequency ``i`` gets one keep/prune decision shared by the real
    channel, the imaginary channel, and every head at once. (A coarser
    ``granularity=head_dim`` only enforces the first bullet, not the second.)

    Returns the length-``head_dim//2`` boolean pattern (True = keep that
    rotary frequency). Raises ``ValueError`` if either requirement is broken.
    """
    if keep_mask.numel() != num_heads * head_dim:
        raise ValueError(
            f"{source}: rope_prune_mask has {keep_mask.numel()} entries, expected "
            f"num_heads * head_dim = {num_heads} * {head_dim} = {num_heads * head_dim}."
        )

    pattern = keep_mask.view(num_heads, head_dim)
    if not torch.all(pattern == pattern[0:1]):
        raise ValueError(
            f"{source}: prunable RoPE requires the same keep pattern in every head "
            "(structured_pruning with granularity=head_dim // 2 and repeated_pattern=True); "
            "got a mask that differs across heads."
        )
    pattern = pattern[0]

    half = head_dim // 2
    real, imag = pattern[:half], pattern[half:]
    if not torch.equal(real, imag):
        raise ValueError(
            f"{source}: prunable RoPE requires each rotary frequency's real/imaginary "
            "channel pair (indices i and i + head_dim//2) to be kept or pruned together "
            "(structured_pruning with granularity=head_dim // 2 and repeated_pattern=True); "
            "got a mask that splits a pair."
        )
    return real


def build_rope_prune_projection(
        q_keep_mask: torch.Tensor,
        k_keep_mask: torch.Tensor,
        head_dim: int,
        q_num_heads: int,
        k_num_heads: int,
        ):
    """Build the static 0/1 selection matrix that projects full-size RoPE
    ``cos``/``sin`` (``head_dim//2`` entries) down to the rotary frequencies
    surviving structured pruning of q_proj/k_proj.

    q_proj and k_proj are coupled (``pruning.position_linked``) and must have
    been pruned with a shared mask, so both reduce to the identical
    length-``head_dim//2`` pattern; this is checked explicitly here.

    Returns a ``(kept_freqs, head_dim//2)`` float tensor with exactly one 1
    per row (selecting the corresponding kept frequency); ``cos_pruned = cos
    @ proj.t()`` (and likewise for ``sin``).
    """
    q_pattern = _derive_rope_prune_pattern(q_keep_mask, head_dim, q_num_heads, source="q_proj")
    k_pattern = _derive_rope_prune_pattern(k_keep_mask, head_dim, k_num_heads, source="k_proj")
    if not torch.equal(q_pattern, k_pattern):
        raise ValueError(
            "Prunable RoPE requires q_proj and k_proj to be pruned with the same "
            "shared mask (they are coupled via pruning.position_linked / "
            "pruning.coupled_masks -- use manager.auto_groups() + share_mask=True), "
            "so a single cos/sin projection is valid for both. Got two different "
            "rotary keep patterns."
        )

    kept_idx = torch.nonzero(q_pattern, as_tuple=True)[0]
    proj = torch.zeros(kept_idx.numel(), q_pattern.numel(), dtype=torch.float32)
    proj[torch.arange(kept_idx.numel()), kept_idx] = 1.0
    return proj


__all__ = [
    "precompute_mrope_inv_freqs",
    "precompute_mrope_cos_sin_half",
    "precompute_rope_inv_freqs",
    "precompute_rope_cos_sin_half",
    "apply_rope_multihead",
    "build_rope_prune_projection",
]
