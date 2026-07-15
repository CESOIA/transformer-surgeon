import torch
from typing import List, Optional, Union

# Single source of truth for the ratio -> kept-neuron conversion lives in
# ``blocks`` (the lowest layer) so both the compression algorithms and the
# converted-graph blocks agree on pruned shapes. Re-exported here for callers
# that import it from the pruning methods package.
from ...blocks.pruning_dims import effective_num_pruned, effective_out_features


def build_structured_mask(
    scores: torch.Tensor,
    ratio: float = 0.0,
    granularity: Union[str, int] = "layer",
) -> torch.Tensor:
    """Build a 1-D boolean keep-mask over output rows (``True`` = keep row).

    The number of pruned rows is dictated by :func:`effective_num_pruned`, so the
    ratio -> effective-dim contract stays identical everywhere. With an integer
    ``granularity`` the same number of rows is pruned inside each consecutive
    chunk (per-head-uniform pruning).
    """
    device = scores.device
    dim = scores.size(0)

    if ratio is None or ratio <= 0.0:
        return torch.ones(dim, dtype=torch.bool, device=device)
    if ratio >= 1.0:
        return torch.zeros(dim, dtype=torch.bool, device=device)

    if granularity in ("layer", None):
        num_to_prune = effective_num_pruned(dim, ratio, granularity)
        if num_to_prune == 0:
            return torch.ones(dim, dtype=torch.bool, device=device)
        mask = torch.ones(dim, dtype=torch.bool, device=device)
        indices = torch.topk(scores, num_to_prune, largest=False, sorted=False).indices
        mask[indices] = False
        return mask

    # Per-chunk (e.g. per-head) pruning.
    g = int(granularity)
    if g <= 0:
        raise ValueError(f"granularity must be a positive int or 'layer', got {granularity!r}.")
    if dim % g != 0:
        raise ValueError(f"granularity {g} must evenly divide the output dimension {dim}.")

    per_chunk = int(ratio * g)
    mask = torch.ones(dim, dtype=torch.bool, device=device)
    if per_chunk == 0:
        return mask

    num_chunks = dim // g
    chunked = scores.view(num_chunks, g)
    local_idx = torch.topk(chunked, per_chunk, dim=1, largest=False, sorted=False).indices
    offsets = (torch.arange(num_chunks, device=device) * g).unsqueeze(1)
    flat_idx = (local_idx + offsets).reshape(-1)
    mask[flat_idx] = False
    return mask


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
      :func:`tile_pattern_mask`) prunes the *same* position in every group —
      which is what lets layers with a *different* number of groups (e.g. GQA
      q_proj with 14 heads and k_proj with 2 heads) share one mask.
    * ``repeat=N``: groups are chunked into consecutive runs of ``N``; each
      chunk is reduced independently, producing ``num_groups // N``
      length-``granularity`` patterns concatenated into one flat vector of
      length ``(num_groups // N) * granularity``. ``num_groups`` must be an
      exact multiple of ``N``.

    Model-agnostic: ``granularity`` is just an integer pattern size.
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
                f"({num_groups}=dim/granularity)."
            )
        grouped = grouped.view(num_groups // n, n, g)
        reduce_dim = 1
    else:
        reduce_dim = 0
    if op == "add":
        reduced = grouped.sum(dim=reduce_dim)
    elif op == "multiply":
        reduced = grouped.prod(dim=reduce_dim)
    else:
        raise ValueError(f"Unsupported reduce_op '{op}'. Supported: 'add', 'multiply'.")
    return reduced.reshape(-1)


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
      ``reduce_pattern_scores(..., repeat=N)``); each chunk is repeated ``N``
      times consecutively (``chunk chunk ... | next_chunk next_chunk ...``),
      producing ``num_chunks * N * granularity`` rows, which must equal
      ``out_dim``.
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


def reduce_scores(score_list: List[torch.Tensor], op: Optional[str]) -> torch.Tensor:
    """Reduce a list of per-layer score vectors into one, for shared-mask groups.

    ``op`` must be ``"add"`` or ``"multiply"``; ``None``/``""`` raises (a group
    that shares a mask needs an explicit reduction rule).
    """
    if op in (None, ""):
        raise ValueError(
            "reduce_op is required when sharing a mask across a group. "
            "Set reduce_op to 'add' or 'multiply'."
        )
    if len(score_list) == 0:
        raise ValueError("reduce_scores received an empty score list.")

    result = score_list[0].clone()
    for scores in score_list[1:]:
        if op == "add":
            result = result + scores
        elif op == "multiply":
            result = result * scores
        else:
            raise ValueError(f"Unsupported reduce_op '{op}'. Supported: 'add', 'multiply'.")
    return result


__all__ = [
    "build_structured_mask",
    "effective_num_pruned",
    "effective_out_features",
    "reduce_scores",
    "reduce_pattern_scores",
    "tile_pattern_mask",
]
