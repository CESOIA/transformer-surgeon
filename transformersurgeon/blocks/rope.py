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
        num_kv_groups: int = 1,
        source: str = "",
        ):
    """Extract the **per-kv-group** rotary keep-patterns from a structured-pruning
    keep mask over a q/k projection's full output rows.

    Under GQA each of the ``num_kv_groups`` key/value heads is shared by
    ``group_size = num_heads // num_kv_groups`` query heads; a group's q heads and
    its k head must keep the same rotary frequencies (so their q.k product stays
    aligned), but *different* groups may keep different frequencies. Prunable RoPE
    then requires, per group:
      * the same keep pattern in every head of the group (one projection per group);
      * each frequency's real/imaginary channel pair (indices ``i`` and
        ``i + head_dim//2`` within a head, see ``apply_rope_multihead``) kept or
        pruned together, since they share one ``cos``/``sin`` value;
    and, across groups, the **same kept count** (so the pruned q/k stay rectangular).

    These are produced by ``structured_pruning`` with an integer ``granularity``
    (head_dim) and ``repeated_pattern="auto"`` on the shared q/k group (the
    position_linked path ties real/imag). ``num_kv_groups=1`` recovers the old
    uniform-across-all-heads behaviour.

    Returns a ``(num_kv_groups, head_dim//2)`` boolean tensor (True = keep that
    rotary frequency for that group). Raises ``ValueError`` if any requirement is
    broken.
    """
    if keep_mask.numel() != num_heads * head_dim:
        raise ValueError(
            f"{source}: rope_prune_mask has {keep_mask.numel()} entries, expected "
            f"num_heads * head_dim = {num_heads} * {head_dim} = {num_heads * head_dim}."
        )
    if num_kv_groups <= 0 or num_heads % num_kv_groups != 0:
        raise ValueError(
            f"{source}: num_kv_groups ({num_kv_groups}) must be a positive divisor of "
            f"num_heads ({num_heads})."
        )

    group_size = num_heads // num_kv_groups
    half = head_dim // 2
    # (num_kv_groups, group_size, head_dim)
    grouped = keep_mask.view(num_kv_groups, group_size, head_dim)

    patterns = []
    for g in range(num_kv_groups):
        group = grouped[g]
        if not torch.all(group == group[0:1]):
            raise ValueError(
                f"{source}: prunable RoPE requires the same keep pattern across the heads "
                f"of each kv-group; group {g} differs across its {group_size} head(s) "
                "(structured_pruning with granularity=head_dim and repeated_pattern='auto')."
            )
        head = group[0]
        real, imag = head[:half], head[half:]
        if not torch.equal(real, imag):
            raise ValueError(
                f"{source}: prunable RoPE requires each rotary frequency's real/imaginary "
                "channel pair (indices i and i + head_dim//2) to be kept or pruned together; "
                f"kv-group {g} splits a pair (use repeated_pattern='auto' on the "
                "position_linked q/k group, which ties them)."
            )
        patterns.append(real)

    patterns = torch.stack(patterns, dim=0)  # (num_kv_groups, half)
    kept_per_group = patterns.sum(dim=1)
    if not torch.all(kept_per_group == kept_per_group[0]):
        raise ValueError(
            f"{source}: prunable RoPE requires every kv-group to keep the same number of "
            f"rotary frequencies (so the pruned q/k stay rectangular); got per-group kept "
            f"counts {kept_per_group.tolist()}."
        )
    return patterns


def build_rope_prune_projection(
        q_keep_mask: torch.Tensor,
        k_keep_mask: torch.Tensor,
        head_dim: int,
        q_num_heads: int,
        k_num_heads: int,
        ):
    """Build the static 0/1 selection matrices that project full-size RoPE
    ``cos``/``sin`` (``head_dim//2`` entries) down to the rotary frequencies
    surviving structured pruning of q_proj/k_proj, **one per kv-group**.

    q_proj and k_proj are coupled (``pruning.position_linked``) and pruned with a
    shared ``repeated_pattern='auto'`` mask, so within each kv-group the q heads
    and their k head keep the same frequencies (checked here); different groups
    may keep different frequencies, but all groups keep the same count.

    Returns a ``(k_num_heads, kept_freqs, head_dim//2)`` float tensor (one
    selection matrix per kv-group, each row a one-hot picking a kept frequency).
    ``_project_rope`` applies group ``g`` to ``cos``/``sin`` and broadcasts the
    result to that group's query heads. ``k_num_heads == 1`` (or a fully uniform
    mask) recovers the single-projection behaviour.
    """
    # The number of kv-groups is k_proj's head count; each is shared by
    # q_num_heads // k_num_heads query heads.
    num_kv_groups = k_num_heads
    q_patterns = _derive_rope_prune_pattern(
        q_keep_mask, head_dim, q_num_heads, num_kv_groups, source="q_proj"
    )
    k_patterns = _derive_rope_prune_pattern(
        k_keep_mask, head_dim, k_num_heads, num_kv_groups, source="k_proj"
    )
    if not torch.equal(q_patterns, k_patterns):
        raise ValueError(
            "Prunable RoPE requires each kv-group's q heads and its k head to keep the "
            "same rotary frequencies (they are coupled via pruning.position_linked / "
            "pruning.coupled_masks -- use manager.auto_groups() + share_mask=True with "
            "repeated_pattern='auto'). Got q/k per-kv-group patterns that differ."
        )

    # One (kept_per_group, head_dim//2) 0/1 selection matrix per kv-group, stacked
    # into (num_kv_groups, kept_per_group, head_dim//2). ``cos``/``sin`` (head_dim//2)
    # are projected per group and then broadcast to that group's heads.
    half = head_dim // 2
    kept_per_group = int(q_patterns[0].sum().item())
    proj = torch.zeros(num_kv_groups, kept_per_group, half, dtype=torch.float32)
    for g in range(num_kv_groups):
        kept_idx = torch.nonzero(q_patterns[g], as_tuple=True)[0]
        proj[g, torch.arange(kept_idx.numel()), kept_idx] = 1.0
    return proj


__all__ = [
    "precompute_mrope_inv_freqs",
    "precompute_mrope_cos_sin_half",
    "precompute_rope_inv_freqs",
    "precompute_rope_cos_sin_half",
    "apply_rope_multihead",
    "build_rope_prune_projection",
]
