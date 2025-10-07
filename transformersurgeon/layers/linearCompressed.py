import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union
import math

# Custom linear layer for low-rank decomposition

class LinearCompressed(nn.Linear):
    """
    Linear layer with low-rank decomposition and optional structured pruning.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias. Default: False.
        lrd_rank (int or str): Rank for low-rank decomposition. Use a positive integer for the rank or "full" for no decomposition. Default: "full".

    Note: when low-rank decomposition is used, an additional weight matrix is created internally.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 lrd_rank: Union[int, str] = "full"):
        
        self.prune_mask = None  # To be set externally if needed

        self.lrd_rank = self._check_rank(lrd_rank)
                
        if out_features <= 0:
            # If out_features is 0, skip the layer (the block has been fully pruned)
            self.skip = True
        else:
            self.skip = False
            in_features_eff = self.lrd_rank if isinstance(self.lrd_rank, int) else in_features
            super().__init__(in_features=in_features_eff,
                             out_features=out_features,
                             bias=bias,
                             device=device,
                             dtype=dtype)
            
            # When using low-rank decomposition, create weight_2
            if isinstance(self.lrd_rank, int):
                self.weight_2 = nn.Parameter(torch.empty((in_features, self.lrd_rank), device=device, dtype=dtype))
                nn.init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Skip the layer if out_features is 0
        if self.skip:
            return input
        # Perform with low-rank decomposition
        if isinstance(self.lrd_rank, int):
            weightA = self.weight # shape: (out_features, lrd_rank)
            weightBT = self.weight_2 # shape: (in_features, lrd_rank)

            if self.prune_mask is not None: # Apply soft structured pruning if mask is available
                weightA = weightA * self.prune_mask

            # Compute: x @ (weightA @ weightB).T + bias which is the same as x @ weightB.T @ weightA.T + bias
            return F.linear(input @ weightBT, weightA, self.bias)
        elif self.lrd_rank == "full": # Perform normally if rank is full
            # Apply soft structured pruning if mask is available
            if self.prune_mask is not None:
                weight = weight * self.prune_mask

            return F.linear(input, self.weight, self.bias)
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