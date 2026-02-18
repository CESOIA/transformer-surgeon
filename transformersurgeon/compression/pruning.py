import torch
from .abstract import Compressor

def _unstructured_mask(
    weight,
    criterion="magnitude",
    pruning_ratio=0.0,
    reverse=False,
    ) -> torch.Tensor:
    """
    Returns a mask for unstructured pruning based on the specified criterion.
    This function generates a binary mask that can be applied to the weight tensor
    to zero out individual weights (unstructured pruning) based on their magnitude.

    Args:
        weight (torch.Tensor): The weight tensor to be pruned.
        criterion (str): The criterion to use for calculating the importance of weights. Default is "magnitude". Supported criteria: "magnitude", "gradient", "random".
        pruning_ratio (float): The ratio of weights to prune (between 0 and 1). Default is 0.0 (no pruning).
        reverse (bool): If True, prunes the most important weights instead of the least important ones. Default is False.

    Returns:
        torch.Tensor: A binary mask with the same shape as the weight tensor.
    """        
    if pruning_ratio >= 1.0: # Prune all weights
        return torch.zeros_like(weight, dtype=torch.bool, device=weight.device)
    
    # Determine the number of weights to prune
    num_weights_to_prune = int(pruning_ratio * weight.numel())
    
    if pruning_ratio <= 0.0 or num_weights_to_prune == 0: # No pruning, return a mask of all ones
        return torch.ones_like(weight, dtype=torch.bool, device=weight.device)
    
    # Calculate importance scores based on the criterion
    if criterion == "magnitude":
        scores = torch.abs(weight)
    elif criterion == "gradient":
        if weight.grad is None:
            raise ValueError("Gradient is required for gradient-based scoring but is not available.")
        scores = torch.abs(weight * weight.grad) # Element-wise product of weights and their gradients
    elif criterion == "random":
        scores = torch.rand_like(weight)
    else:
        raise ValueError(f"Unsupported criterion '{criterion}' for unstructured pruning.")
    
    # Generate the mask by selecting the top-k scores (either the smallest or largest based on the reverse flag)
    threshold = torch.topk(scores.flatten(), num_weights_to_prune, largest=reverse, sorted=False).values.max()
    mask = scores < threshold if reverse else scores > threshold
    
    return mask.view_as(weight)

def _structured_mask(
    weight,
    norm=2,
    criterion="magnitude",
    pruning_ratio=0.0,
    reverse=False,
    ) -> torch.Tensor:
    """
    Returns a mask for structured pruning based on the specified norm.
    This function generates a binary mask that can be applied to the weight tensor
    to zero out entire rows (structured pruning) based on their L2 norm.

    Args:
        weight (torch.Tensor): The weight tensor to be pruned.
        norm (int): The norm to use for calculating the importance of rows. Default is 2 (L2 norm).
        criterion (str): The criterion to use for calculating the importance of rows. Default is "magnitude". Supported criteria: "magnitude", "gradient", "random".
        pruning_ratio (float): The ratio of rows to prune (between 0 and 1). Default is 0.0 (no pruning).
        reverse (bool): If True, prunes the most important rows instead of the least important ones. Default is False.

    Returns:
        torch.Tensor: A binary mask vector with the same number of elements as the rows of the weight tensor.
        kept (int): Number of rows kept after pruning.
    """
    current_rows = weight.size(0)
    
    if pruning_ratio >= 1.0: # Prune all rows
        return torch.zeros(weight.size(0), dtype=torch.bool, device=weight.device), 0
    
    # Determine the number of rows to prune
    num_rows_to_prune = int(pruning_ratio * current_rows)
    kept = current_rows - num_rows_to_prune
    
    if pruning_ratio <= 0.0 or num_rows_to_prune == 0: # Prune no rows
        return torch.ones(weight.size(0), dtype=torch.bool, device=weight.device), weight.size(0)
    
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

    # Generate the mask by selecting the top-k scores (either the smallest or largest based on the reverse flag)
    threshold = torch.topk(scores, num_rows_to_prune, largest=reverse, sorted=False).values.max()
    mask = scores < threshold if reverse else scores > threshold
    
    return mask, kept

class Pruner(Compressor):
    def __init__(
        self,
        ratio=0.0,
        mode="structured",
        criterion="magnitude"
        ):
        # Configuration
        self.ratio = ratio
        self.mode = mode
        self.criterion = criterion
        # Helper objects for applying and restoring pruning
        self.prune_mask = None
        self.kept = None

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return # No compression needed based on the configuration

        # Extract pruning configuration
        ratio = self.ratio
        mode = self.mode
        criterion = self.criterion

        if ratio > 0:
            if not soft_applied:
                if mode == 'structured':
                    prune_mask, kept = _structured_mask(
                        module.weight.data,
                        norm=2,
                        criterion=criterion,
                        pruning_ratio=ratio)
                    self.prune_mask = prune_mask.cpu() # store on CPU to save GPU memory
                    self.kept = kept # store for later use
                elif mode == 'unstructured':
                    prune_mask = _unstructured_mask(
                        module.weight.data,
                        criterion=criterion,
                        pruning_ratio=ratio)
                    self.prune_mask = prune_mask.cpu() # store on CPU to save GPU memory
                else:
                    raise ValueError(f"Unsupported pruning mode '{mode}'.")
                
                if not hard:
                    # Apply soft pruning by zeroing out the pruned rows/weights
                    if mode == 'structured':
                        module.weight.data = module.weight.data*prune_mask.unsqueeze(1).to(torch.float32)
                    else:
                        module.weight.data = module.weight.data*prune_mask.to(torch.float32)
            
            if hard:
                if mode == 'unstructured':
                    raise ValueError("Hard pruning is not supported for unstructured pruning.")
                
                # Apply hard pruning by removing the pruned rows
                module.weight.data = module.weight.data[self.prune_mask.to(module.weight.device)]
                if module.bias is not None:
                    module.bias.data = module.bias.data[self.prune_mask.to(module.bias.device)] 
                module.out_features = self.kept
                del self.prune_mask

    def restore(self, module):
        self.prune_mask = None
        self.kept = None

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

__all__ = [
    "Pruner",
    "validate_pruning_ratio",
    "validate_pruning_mode",
    "validate_pruning_criterion"
]