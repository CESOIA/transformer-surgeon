import torch

def _reshape_for_granularity(scores: torch.Tensor, pruning_ratio: int, granularity: str):
    if granularity == "layer":
        num_to_prune = int(round(pruning_ratio * scores.numel()))
        num_to_keep = scores.numel() - num_to_prune
        scores_view = scores.view(-1).unsqueeze(0)
    elif granularity == "neuron":
        num_to_prune = int(round(pruning_ratio * scores.size(1)))
        num_to_keep = scores.size(1) - num_to_prune
        scores_view = scores
    elif isinstance(granularity, int) and granularity > 0:
        num_to_prune = int(round(granularity * pruning_ratio))
        num_to_keep = granularity - num_to_prune
        scores_view = scores.view(-1, granularity)
    else:
        raise ValueError(f"Unsupported granularity '{granularity}' for unstructured pruning.")
    return scores_view, num_to_prune, num_to_keep

def build_unstructured_mask(
    scores: torch.Tensor,
    pruning_ratio: float = 0.0,
    granularity: str = "layer",
) -> torch.Tensor:
    
    device = scores.device

    if pruning_ratio >= 1.0:
        return torch.zeros_like(scores, dtype=torch.bool, device=device)
    
    scores_view, num_to_prune, num_to_keep = _reshape_for_granularity(scores, pruning_ratio, granularity)

    if pruning_ratio <= 0.0 or num_to_prune == 0:
        return torch.ones_like(scores, dtype=torch.bool, device=device)

    if num_to_prune < num_to_keep:
        indices = torch.topk(scores_view, num_to_prune, largest=False, sorted=False, dim=-1).indices
        row_ids = torch.arange(indices.size(0), device=device)[:, None]
        mask = torch.ones_like(scores_view, dtype=torch.bool)
        mask[row_ids, indices] = False
    else:
        indices = torch.topk(scores_view, num_to_keep, largest=True, sorted=False, dim=-1).indices
        row_ids = torch.arange(indices.size(0), device=device)[:, None]
        mask = torch.zeros_like(scores_view, dtype=torch.bool)
        mask[row_ids, indices] = True

    return mask.view_as(scores)


__all__ = ["build_unstructured_mask"]