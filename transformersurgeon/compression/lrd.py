import torch
from typing import Union
from .abstract import Compressor

def _low_rank_svd(weight, rank: int) -> torch.Tensor:
    """
    Performs low-rank decomposition on the given weight matrix using SVD.

    Args:
        weight (torch.Tensor): The weight matrix to be decomposed.
        rank (int): The target rank for the decomposition.

    Returns:
        torch.Tensor: The first matrix of the low-rank decomposition.
        torch.Tensor: The second matrix of the low-rank decomposition.
    """
    if rank >= min(weight.size()):
        # No decomposition possible, launch error
        raise ValueError(f"Rank {rank} must be less than the minimum dimension of the weight matrix ({weight.size(0), weight.size(1)}).")
    
    # Perform SVD
    weight_f32 = weight.float() # Convert to float32 for SVD computation
    U, S, Vh = torch.linalg.svd(weight_f32, full_matrices=False)

    # Keep only the top 'rank' components
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    # Reconstruct the low-rank approximation
    US_r_out = (U_r * S_r.unsqueeze(0)).to(weight.dtype)
    V_r_out = Vh_r.to(weight.dtype)
    return US_r_out.contiguous(), V_r_out.contiguous()

class LRDer(Compressor):
    def __init__(
        self,
        config,
        ):
        # Configuration
        self.config = config
        # Local temporary configuration
        self.rank = self.config["rank"]

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return # No compression needed based on the configuration

        # Extract temp configuration
        rank = self.rank
        # Apply temp configuration to module config
        self.config["rank"] = rank
    
        if rank:
            if not soft_applied:
                with torch.no_grad():
                    US_r, V_r = _low_rank_svd(module.weight.detach(), rank)
                    module.init_lrd(rank)
                    module.weight[:, :rank].copy_(US_r)
                    module.weight_2[:rank, :].copy_(V_r)

    def restore(self, module):
        if not self._to_compress():
            # Restore module configuration
            self.config["rank"] = "full"

        if not hasattr(module, 'weight_2'):
            raise AttributeError("Module does not have 'weight_2' attribute required for LRD restoration.")
        
        # Combine the low-rank components to restore the original weight matrix
        with torch.no_grad():
            restored_weight = module.weight.detach() @ module.weight_2.detach()
            module.cancel_lrd()
            module.weight.copy_(restored_weight)
        
    def _to_compress(self):
        # Check if LRD has to be applied based on the rank configuration
        return self.rank != "full"
    
    def __repr__(self):
        string = f"LRDer(rank={self.rank})"
        return string

# Configuration validators
        
def validate_lrd_rank(rank: Union[int, str]) -> None:
    if rank is not None:
        if isinstance(rank, str):
            if rank != "full":
                raise ValueError(f"LRD rank must be 'full' or a positive integer, but got '{rank}'.")
        elif isinstance(rank, int):
            if rank <= 0:
                raise ValueError(f"LRD rank must be a positive integer, but got {rank}.")
        else:
            raise ValueError(f"LRD rank must be 'full' or a positive integer, but got type {type(rank)}.")

__all__ = [
    "LRDer",
    "validate_lrd_rank"
]