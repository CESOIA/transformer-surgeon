import torch

def build_structured_mask(
    scores: torch.Tensor,
    pruning_ratio: float = 0.0,
) -> torch.Tensor:
    device = scores.device
    current_rows = scores.size(0)

    if pruning_ratio >= 1.0:
        return torch.zeros(current_rows, dtype=torch.bool, device=device)

    num_to_prune = int(pruning_ratio * current_rows)
    num_to_keep = current_rows - num_to_prune

    if pruning_ratio <= 0.0 or num_to_prune == 0:
        return torch.ones(current_rows, dtype=torch.bool, device=device)

    if num_to_prune < num_to_keep:
        indices = torch.topk(scores, num_to_prune, largest=False, sorted=False).indices
        mask = torch.ones(current_rows, dtype=torch.bool, device=device)
        mask[indices] = False
    else:
        indices = torch.topk(scores, num_to_keep, largest=True, sorted=False).indices
        mask = torch.zeros(current_rows, dtype=torch.bool, device=device)
        mask[indices] = True

    return mask


__all__ = ["build_structured_mask"]