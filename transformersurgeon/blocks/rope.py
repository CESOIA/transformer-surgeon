import math
import warnings

import torch


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
    if params["rope_type"] == "mrope":  # legacy Qwen2-VL marker: default frequencies, sectioned (HF does the same)
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


def _inv_freq_default(head_dim: int, params: dict, device="cpu") -> torch.Tensor:
    return precompute_rope_inv_freqs(head_dim=head_dim, base=float(params.get("rope_theta", 1e6)), device=device)


def _inv_freq_llama3(head_dim: int, params: dict, device="cpu") -> torch.Tensor:
    """Llama 3.x frequency scaling (HF ``_compute_llama3_parameters``): low
    frequencies (wavelength above ``original_max_position_embeddings /
    low_freq_factor``) are divided by ``factor``, high frequencies are kept, and
    the band in between is interpolated smoothly. The attention factor is 1."""
    for key in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"):
        if key not in params:
            raise ValueError(f"RoPE type 'llama3' needs '{key}' in the config's RoPE parameters.")
    inv_freq = _inv_freq_default(head_dim, params, device)
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


# rope_type -> inverse-frequency function (head_dim, params, device). Every entry
# must keep the attention scaling at 1 (cos/sin are not rescaled by the decoder).
ROPE_INV_FREQ = {
    "default": _inv_freq_default,
    "llama3": _inv_freq_llama3,
}


def resolve_rope_type(config, indexing_rope_type: str = "default") -> str:
    """RoPE variant a converted decoder must use.

    ``indexing_rope_type`` comes from the model's indexing file: a fixed type
    (e.g. ``"default"`` for Qwen2) or ``"auto"`` to take the config's
    ``rope_type`` (Llama: ``"llama3"`` for Llama 3.x, ``"default"`` for Llama
    2 / TinyLlama). A fixed type that disagrees with a scaled config, or a type
    without an implementation, raises instead of silently using plain RoPE.
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
    if rope_type not in ROPE_INV_FREQ:
        raise NotImplementedError(
            f"RoPE type '{rope_type}' is not implemented (available: {sorted(ROPE_INV_FREQ)})."
        )
    return rope_type


def precompute_rope_inv_freqs_for(rope_type: str, head_dim: int, params: dict, device="cpu") -> torch.Tensor:
    """Inverse frequencies ``(head_dim//2,)`` of one registered RoPE variant."""
    if rope_type not in ROPE_INV_FREQ:
        raise NotImplementedError(
            f"RoPE type '{rope_type}' is not implemented (available: {sorted(ROPE_INV_FREQ)})."
        )
    return ROPE_INV_FREQ[rope_type](head_dim, params, device)


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
    "ROPE_INV_FREQ",
    "rope_parameters_from_config",
    "rope_theta_from_config",
    "resolve_rope_type",
    "precompute_rope_inv_freqs_for",
    "precompute_mrope_inv_freqs",
    "precompute_mrope_cos_sin_half",
    "precompute_rope_inv_freqs",
    "precompute_rope_cos_sin_half",
    "apply_rope_multihead",
    "build_rope_prune_projection",
]
