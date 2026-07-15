"""Repeated-pattern replicate logic for structured pruning.

Kept out of ``mask_generation`` on purpose: ``mask_generation`` only turns an
already-correctly-shaped score vector into a keep-mask, while the *replicate*
decisions -- how a per-neuron score vector is collapsed into one or more
repeated patterns, and how a pattern mask is tiled back up to the full output
dimension -- live here and are orchestrated by ``StructuredPruner`` (which knows
about groups, siblings, and per-layer head structure).

All functions are model-agnostic: ``granularity`` is just an integer pattern
size (e.g. an attention ``head_dim`` or ``head_dim // 2``).
"""

import torch
from typing import Optional


def reduce_pattern_scores(
    scores: torch.Tensor,
    granularity: int,
    op: Optional[str],
    repeat: Optional[int] = None,
) -> torch.Tensor:
    """Collapse a per-neuron score vector into one or more repeated-pattern vectors.

    The output dimension is treated as ``num_groups`` repetitions of a pattern of
    size ``granularity`` (e.g. attention heads, each of size ``head_dim``).

    * ``repeat=None`` (default): the scores of the corresponding position across
      *every* group are reduced with ``op`` into one length-``granularity``
      vector. Building a mask from this and tiling it back (see
      :func:`tile_pattern_mask`) prunes the *same* position in every group --
      which is what lets layers with a *different* number of groups (e.g. GQA
      q_proj with 14 heads and k_proj with 2 heads) share one mask.
    * ``repeat=N``: groups are chunked into consecutive runs of ``N``; each
      chunk is reduced independently, producing ``num_groups // N``
      length-``granularity`` patterns concatenated into one flat vector of
      length ``(num_groups // N) * granularity``. ``num_groups`` must be an
      exact multiple of ``N``.
    """
    dim = scores.size(0)
    g = int(granularity)
    if g <= 0 or dim % g != 0:
        raise ValueError(
            f"repeated_pattern requires granularity to divide the output dim; "
            f"got dim={dim}, granularity={granularity}."
        )
    if op in (None, ""):
        raise ValueError(
            "reduce_op is required for repeated_pattern scoring. Set 'add' or 'multiply'."
        )
    num_groups = dim // g
    grouped = scores.view(num_groups, g)
    if repeat is not None:
        n = int(repeat)
        if n <= 0 or num_groups % n != 0:
            raise ValueError(
                f"repeated_pattern={n} must evenly divide the number of groups "
                f"(num_groups=dim/granularity <-> {num_groups}={dim}/{granularity})."
            )
        grouped = grouped.view(num_groups // n, n, g)
        reduce_dim = 1
    else:
        reduce_dim = 0
    return _reduce(grouped, op, reduce_dim).reshape(-1)


def reduce_member_to_patterns(
    scores: torch.Tensor,
    granularity: int,
    num_patterns: int,
    op: Optional[str],
    target_gpp: int,
) -> torch.Tensor:
    """Reduce one shared-mask member's scores into ``num_patterns`` patterns.

    Used by the auto-derived GQA path: a member with ``num_groups`` groups of
    size ``granularity`` is collapsed into ``num_patterns`` length-``granularity``
    patterns (``num_groups`` must be a multiple of ``num_patterns``), summing
    over its ``num_groups // num_patterns`` groups-per-pattern.

    To weight every member of the group equally regardless of how many groups it
    has (the user's "duplicate scoring matrix" step), each member's groups are
    first *duplicated* up to ``target_gpp`` groups-per-pattern before reducing --
    so e.g. a GQA k_proj (1 group per pattern) contributes with the same weight
    as q_proj (``group_size`` groups per pattern). Returns a flat ``[num_patterns
    * granularity]`` vector.
    """
    dim = scores.size(0)
    g = int(granularity)
    P = int(num_patterns)
    if g <= 0 or dim % g != 0:
        raise ValueError(
            f"repeated_pattern='auto' requires granularity to divide the output dim; "
            f"got dim={dim}, granularity={granularity}."
        )
    num_groups = dim // g
    if P <= 0 or num_groups % P != 0:
        raise ValueError(
            f"repeated_pattern='auto' requires the pattern count {P} to divide the "
            f"number of groups (num_groups=dim/granularity <-> {num_groups}={dim}/{g})."
        )
    if op in (None, ""):
        raise ValueError(
            "reduce_op is required for repeated_pattern='auto'. Set 'add' or 'multiply'."
        )
    gpp = num_groups // P
    t = int(target_gpp)
    if t % gpp != 0:
        raise ValueError(
            f"target groups-per-pattern {t} must be a multiple of this member's "
            f"groups-per-pattern {gpp}."
        )
    # [P, gpp, g] -> duplicate the gpp groups up to target_gpp -> reduce -> [P, g]
    grouped = scores.view(P, gpp, g)
    if t != gpp:
        grouped = grouped.repeat_interleave(t // gpp, dim=1)
    return _reduce(grouped, op, reduce_dim=1).reshape(-1)


def tile_pattern_mask(
    pattern_mask: torch.Tensor,
    out_dim: int,
    granularity: Optional[int] = None,
    repeat: Optional[int] = None,
) -> torch.Tensor:
    """Tile pattern keep-mask(s) up to ``out_dim`` rows.

    * ``repeat=None`` (default): ``pattern_mask`` is a single length-``granularity``
      pattern; it's repeated ``out_dim // g`` times (``out_dim`` must be a whole
      number of pattern repetitions).
    * ``repeat=N``: ``pattern_mask`` holds ``num_chunks`` concatenated
      length-``granularity`` patterns (as produced by
      ``reduce_pattern_scores(..., repeat=N)`` or ``reduce_member_to_patterns``);
      each chunk is repeated ``N`` times consecutively (``chunk chunk ... |
      next_chunk next_chunk ...``), producing ``num_chunks * N * granularity``
      rows, which must equal ``out_dim``.
    """
    if repeat is None:
        g = pattern_mask.numel()
        if g == 0 or out_dim % g != 0:
            raise ValueError(f"pattern mask length {g} must divide out_dim {out_dim}.")
        return pattern_mask.repeat(out_dim // g)

    g = int(granularity)
    n = int(repeat)
    if g <= 0 or pattern_mask.numel() % g != 0:
        raise ValueError(
            f"pattern mask length {pattern_mask.numel()} must be a multiple of "
            f"granularity {g}."
        )
    num_chunks = pattern_mask.numel() // g
    tiled = pattern_mask.view(num_chunks, 1, g).expand(num_chunks, n, g).reshape(-1)
    if tiled.numel() != out_dim:
        raise ValueError(
            f"repeated_pattern={n} tiling produced {tiled.numel()} rows, expected "
            f"out_dim {out_dim}."
        )
    return tiled


def _reduce(grouped: torch.Tensor, op: str, reduce_dim: int) -> torch.Tensor:
    if op == "add":
        return grouped.sum(dim=reduce_dim)
    if op == "multiply":
        return grouped.prod(dim=reduce_dim)
    raise ValueError(f"Unsupported reduce_op '{op}'. Supported: 'add', 'multiply'.")


__all__ = [
    "reduce_pattern_scores",
    "reduce_member_to_patterns",
    "tile_pattern_mask",
]
