"""Static geometry of rotary embeddings whose query/key channels were pruned.

When structured pruning removes channels of the query and key projections, each
group of heads keeps a subset of the rotary frequencies. These helpers turn the
recorded keep-mask into a fixed 0/1 selection matrix that the attention atom
applies to the precomputed cos/sin tables. They decide nothing: the mask is
produced by the compression step beforehand.
"""

import torch


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


__all__ = ["build_rope_prune_projection"]
