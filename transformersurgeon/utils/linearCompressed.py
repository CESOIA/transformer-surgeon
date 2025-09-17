import torch
import torch.nn as nn
from typing import Union

# Custom linear layer for low-rank decomposition

# Note on low-rank decomposition:
#   when using low-rank decomposition, concatenate the two decomposed matrices into a single weight tensor.
#   weight = torch.cat((weightA, weightB.T), dim=0)

class LinearCompressed(nn.Linear):
    def __init__(self, 
                 in_features, 
                 out_features,
                 bias=False,
                 lrd_rank: Union[int, str] = "full"):
        
        self.prune_mask = None  # To be set externally if needed
        self.beta_vcon = None  # To be set externally if needed

        self.lrd_rank = self._check_rank(lrd_rank)
                
        if out_features <= 0:
            # If out_features is 0, skip the layer (the block has been fully pruned)
            self.skip = True
        else:
            self.skip = False
            super().__init__(in_features, out_features, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Skip the layer if out_features is 0
        if self.skip:
            return input
        # Perform with low-rank decomposition
        if isinstance(self.lrd_rank, int):
            weight1 = self.weight[:self.out_features, :]
            weight2 = self.weight[self.out_features:, :]

            if self.prune_mask is not None: # Apply soft structured pruning if mask is available
                weight1 = weight1 * self.prune_mask

            # Compute: x @ weight1 @ weight2.T + bias which is the same as: x @ weight.T + bias
            output = input @ weight2 @ weight1.t()
            if self.bias is not None:
                output += self.bias
            return output
        elif self.lrd_rank == "full": # Perform normally if rank is full
            weight = self.weight

            # Apply soft structured pruning if mask is available
            if self.prune_mask is not None:
                weight = weight * self.prune_mask

            # Compute: x @ weight.T + bias
            output = input @ self.weight.t()
            if self.bias is not None:
                output += self.bias
            return output
        # Manage value errors
        else:
            raise ValueError(f"Unsupported low-rank decomposition value: {self.lrd_rank}")    
        
    def set_lrd_rank(self, rank: Union[int, str]):
        self.lrd_rank = self._check_rank(rank)
        
    def set_prune_mask(self, mask: torch.Tensor):
        if mask.rank != 1 or mask.size(0) != self.out_features:
            raise ValueError("Prune mask must be a 1D tensor with the same size as out_features.")
        self.prune_mask = mask

    def reset_prune_mask(self):
        self.prune_mask = None

    def _check_rank(self, rank: Union[int, str]):
        if isinstance(rank, int):
            if rank < 1:
                raise ValueError("Low-rank decomposition rank must be at least 1.")
        elif rank != "full":
            raise ValueError("Low-rank decomposition rank must be a positive integer or 'full'.")
        
        return rank
        
    def __repr__(self):
        return super().__repr__() + f"(lrd_rank={self.lrd_rank})"

    def __str__(self):
        return self.__repr__()
    
__all__ = ["LinearCompressed"]