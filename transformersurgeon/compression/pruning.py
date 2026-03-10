import torch
from .abstract import Compressor

def _unstructured_mask(
    weight,
    criterion="magnitude",
    granularity="layer",
    pruning_ratio=0.0,
    ) -> torch.Tensor:
    """
    Generate an unstructured pruning mask for the given weight tensor based on the specified criterion and pruning ratio.
    Args:
        weight (torch.Tensor): The weight tensor to be pruned.
        criterion (str): The criterion for scoring weights, one of "magnitude", "gradient", or "random".
        granularity (str): 
        pruning_ratio (float): The ratio of weights to prune, between 0.0 and 1.0.
    """ 
    if pruning_ratio >= 1.0: # Prune all weights
        return torch.zeros_like(weight, dtype=torch.bool, device=weight.device)
    
    # Determine the number of weights to prune based on granularity
    if granularity == "layer": # remove % of weights across the entire layer
        num_to_prune = int(round(pruning_ratio * weight.numel()))
        num_to_keep = weight.numel() - num_to_prune
        weight_reshaped = weight.view(-1).unsqueeze(0) # Flatten the weights for layer-wise pruning
    elif granularity == "neuron": # remove % of weights per output neuron (row-wise)
        num_to_prune = int(round(pruning_ratio * weight.size(1)))
        num_to_keep = weight.size(1) - num_to_prune
        weight_reshaped = weight # Keep original shape for neuron-wise pruning
    elif isinstance(granularity, int) and granularity > 0: # remove % of weights each group of 'granularity' weights
        num_to_prune = int(round(granularity * pruning_ratio))
        num_to_keep = granularity - num_to_prune
        weight_reshaped = weight.view(-1, granularity) # Reshape weights into groups of 'granularity'
    else:
        raise ValueError(f"Unsupported granularity '{granularity}' for unstructured pruning.")
    
    if pruning_ratio <= 0.0 or num_to_prune == 0: # No pruning, return a mask of all ones
        return torch.ones_like(weight, dtype=torch.bool, device=weight.device)
    
    # Calculate importance scores based on the criterion
    if criterion == "magnitude":
        scores = torch.abs(weight_reshaped)
    elif criterion == "gradient":
        if weight.grad is None:
            raise ValueError("Gradient is required for gradient-based scoring but is not available.")
        grad_reshaped = weight.grad.view_as(weight_reshaped)
        scores = torch.abs(weight_reshaped * grad_reshaped) # Element-wise product of weights and their gradients
    elif criterion == "random":
        scores = torch.rand_like(weight_reshaped)
    else:
        raise ValueError(f"Unsupported criterion '{criterion}' for unstructured pruning.")
    
    # Generate the mask by selecting the top-k scores, keeping the length of "indices" to the minimum
    if num_to_prune < num_to_keep:
        indices = torch.topk(scores, num_to_prune, largest=False, sorted=False, dim=-1).indices
        row_ids = torch.arange(indices.size(0), device=weight.device)[:, None]
        mask = torch.ones_like(weight_reshaped, dtype=torch.bool)
        mask[row_ids, indices] = False
    else:
        indices = torch.topk(scores, num_to_keep, largest=True, sorted=False, dim=-1).indices
        row_ids = torch.arange(indices.size(0), device=weight.device)[:, None]
        mask = torch.zeros_like(weight_reshaped, dtype=torch.bool)
        mask[row_ids, indices] = True
    
    return mask.view_as(weight)

def _structured_mask(
    weight,
    norm=2,
    criterion="magnitude",
    pruning_ratio=0.0,
    ) -> torch.Tensor:
    current_rows = weight.size(0)
    
    if pruning_ratio >= 1.0: # Prune all rows
        return torch.zeros(weight.size(0), dtype=torch.bool, device=weight.device)
    
    # Determine the number of rows to prune
    num_to_prune = int(pruning_ratio * current_rows)
    num_to_keep = current_rows - num_to_prune
    
    if pruning_ratio <= 0.0 or num_to_prune == 0: # Prune no rows
        return torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
    
    # Generate scores
    if criterion == "magnitude":
        scores = torch.norm(weight, p=norm, dim=1) # Calculate the norm of each row
    elif criterion == "gradient":
        if weight.grad is None:
            raise ValueError("Gradient is required for gradient-based scoring but is not available.")
        scores = torch.norm(weight*weight.grad, p=norm, dim=1) # Calculate the norm of the gradient of each row
    elif criterion == "random":
        scores = torch.rand(weight.size(0), device=weight.device)
    else:
        raise ValueError(f"Unsupported criterion '{criterion}' for structured pruning.")

    # Generate the mask by selecting the top-k scores, keeping the length of "indices" to the minimum
    if num_to_prune < num_to_keep:
        indices = torch.topk(scores, num_to_prune, largest=False, sorted=False).indices
        mask = torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
        mask[indices] = False
    else:
        indices = torch.topk(scores, num_to_keep, largest=True, sorted=False).indices
        mask = torch.zeros(weight.size(0), dtype=torch.bool, device=weight.device)
        mask[indices] = True
    
    return mask

class Pruner(Compressor):
    def __init__(
        self,
        config,
        ):
        # Configuration
        self.config = config
        # Local temporary configuration
        self.ratio = self.config["ratio"]
        self.mode = self.config["mode"]
        self.criterion = self.config["criterion"]
        self.granularity = self.config["granularity"]

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return # No compression needed based on the configuration

        # Extract temp configuration
        ratio = self.ratio
        mode = self.mode
        criterion = self.criterion
        granularity = self.granularity
        # Apply temp configuration to module config
        self.config["ratio"] = ratio
        self.config["mode"] = mode
        self.config["criterion"] = criterion
        self.config["granularity"] = granularity

        if ratio > 0:
            if not soft_applied:
                if mode == 'structured':
                    prune_mask = _structured_mask(
                        module.weight,
                        norm=2,
                        criterion=criterion,
                        pruning_ratio=ratio)
                elif mode == 'unstructured':
                    prune_mask = _unstructured_mask(
                        module.weight,
                        criterion=criterion,
                        granularity=granularity,
                        pruning_ratio=ratio)
                else:
                    raise ValueError(f"Unsupported pruning mode '{mode}'.")
                
                if not hard:
                    # Apply soft pruning by zeroing out the pruned rows/weights
                    with torch.no_grad():
                        dtype = module.weight.dtype
                        if mode == 'structured':
                            module.weight.mul_(prune_mask.unsqueeze(1).to(dtype))
                            if module.bias is not None:
                                module.bias.mul_(prune_mask.to(dtype))
                        else:
                            module.weight.mul_(prune_mask.to(dtype))
            
            if hard:
                raise NotImplementedError("Hard pruning is not implemented yet.")

    def restore(self, module):
        # Hard pruning is not implemented, no restoration needed
        # Soft pruning does not change topology, so no restoration needed
        # Restore module configuration
        self.config["ratio"] = 0.0
        self.config["mode"] = "structured"
        self.config["criterion"] = "magnitude"
        self.config["granularity"] = "layer"

    def _to_compress(self):
        # Check if pruning has to be applied based on the ratio configuration
        return self.ratio > 0.0
    
    def __repr__(self):
        string = f"Pruner(ratio={self.ratio}, mode='{self.mode}', criterion='{self.criterion}')"
        return string

# Configuration validators

def validate_pruning_ratio(ratio: float) -> None:
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Pruning ratio must be between 0.0 and 1.0, but got {ratio}.")
    
def validate_pruning_mode(mode: str) -> None:
    valid_modes = ["structured", "unstructured"]
    if mode is not None and mode not in valid_modes:
        raise ValueError(f"Pruning mode must be one of {valid_modes}, but got '{mode}'.")
    
def validate_pruning_criterion(criterion: str) -> None:
    valid_criteria = ["magnitude", "gradient", "random"]
    if criterion is not None and criterion not in valid_criteria:
        raise ValueError(f"Pruning criterion must be one of {valid_criteria}, but got '{criterion}'.")

def validate_pruning_granularity(granularity: str) -> None:
    valid_granularities = ["layer", "neuron"]
    if granularity is not None and granularity not in valid_granularities and not (isinstance(granularity, int) and granularity > 0):
        raise ValueError(f"Pruning granularity must be one of {valid_granularities} or a positive integer, but got '{granularity}'.")

__all__ = [
    "Pruner",
    "validate_pruning_ratio",
    "validate_pruning_mode",
    "validate_pruning_criterion",
    "validate_pruning_granularity"
]