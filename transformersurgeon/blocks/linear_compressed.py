import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import Union
import math

# Custom linear layer for low-rank decomposition
class LinearCompressed(nn.Linear):
    """
    Linear layer with support for low-rank decomposition and quantization. This layer can be used to replace standard Linear layers in a transformer model to enable compression techniques such as pruning, low-rank decomposition, and quantization.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias. Default: False.
        rank (Union[int, str]): The target rank for low-rank decomposition. If set to "full", no decomposition is applied. Default: None (equivalent to "full").
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 bias: bool = True,
                 rank = None,
                 device = None,
                 dtype = None,
                 ):

        if out_features <= 0:
            self.skip = True # If out_features is 0, skip the layer (the block has been fully pruned)
            return
        
        self.skip = False
        super().__init__(in_features=in_features,
                            out_features=out_features,
                            bias=bias,
                            device=device,
                            dtype=dtype)
        self.weight_2 = None # Placeholder for the second weight matrix in low-rank decomposition

        self.init_lrd(rank)

    def init_lrd(self, rank):
        # Set rank and initialize weight_2 for low-rank decomposition if needed
        self.rank = "full" if rank is None else rank
        if isinstance(rank, int):
            shape_change = False
            device = self.weight.device
            dtype = self.weight.dtype
            # Initialize weight_2 for low-rank decomposition
            if self.weight_2 is None or self.weight_2.shape[0] != rank:
                self.weight_2 = Parameter(
                    torch.empty([self.rank, self.in_features], device=device, dtype=dtype),
                    requires_grad=True)
                torch.nn.init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
                shape_change = True
            # Adjust weight shape for low-rank decomposition if necessary
            if self.weight.shape[1] != self.rank:
                self.weight = Parameter(
                    torch.empty([self.out_features, self.rank], device=device, dtype=dtype),
                    requires_grad=True)
                torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                shape_change = True
            # Free cache if there was a shape change and we're on GPU
            if shape_change and device.type == "cuda":
                torch.cuda.empty_cache()
            
    def cancel_lrd(self):
        self.rank = "full"
        device = self.weight.device
        dtype = self.weight.dtype
        shape_change = False
        # Remove weight_2 if it exists
        if self.weight_2 is not None:
            self.weight_2 = None
            shape_change = True
        # Revert weight to original shape if it was changed for low-rank decomposition
        if self.weight.shape[1] != self.in_features:
            self.weight = Parameter(
                torch.empty([self.out_features, self.in_features], device=device, dtype=dtype),
                requires_grad=True)
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Free cache if there was a shape change and we're on GPU
        if shape_change and device.type == "cuda":
            torch.cuda.empty_cache()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        # Skip logic
        if self.skip:
            return x

        # LRD logic
        if isinstance(self.rank, int):
            weight = self.weight[:, :self.rank]
            x = F.linear(x, self.weight_2[:self.rank, :])
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)    

    def __str__(self):
        return self.__repr__()
    
__all__ = ["LinearCompressed"]