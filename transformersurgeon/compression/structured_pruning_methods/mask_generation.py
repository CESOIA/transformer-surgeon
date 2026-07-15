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
]
