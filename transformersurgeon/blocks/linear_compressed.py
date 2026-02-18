import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import Union
import math

from ..compression import (
    validate_lrd_rank,
    validate_precision,
    validate_pruning_ratio,
)

# Custom linear layer for low-rank decomposition
class LinearCompressed(nn.Linear):
    """
    Linear layer with support for low-rank decomposition and quantization. This layer can be used to replace standard Linear layers in a transformer model to enable compression techniques such as pruning, low-rank decomposition, and quantization.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias. Default: False.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 bias: bool = True,
                 rank = None,
                 precision = None,
                 qsparsity = None,
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
        
        self.init_soft_quantization(precision, qsparsity)
        self.init_soft_lrd(rank)

    def init_soft_lrd(self, rank):
        device, dtype = self.weight.device, self.weight.dtype
        # Set rank and initialize weight_2 for low-rank decomposition if needed
        self.rank = "full" if rank is None else rank
        validate_lrd_rank(self.rank)
        if isinstance(rank, int) and not hasattr(self, 'weight_2'):
            self.weight_2 = Parameter(
                torch.zeros(
                    (self.in_features, self.in_features),
                    device=device,
                    dtype=dtype),
                requires_grad=True)
            torch.nn.init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))

    def init_soft_quantization(self, precision=None, qsparsity=None):
        # Set precision
        self.precision = "full" if precision is None else precision
        validate_precision(self.precision)

        # Set sparsity and initialize qmask if needed
        self.qsparsity = 0.0 if qsparsity is None else qsparsity
        validate_pruning_ratio(self.qsparsity)
        if self.qsparsity > 0.0 and not hasattr(self, 'qmask'):
            self.register_buffer('qmask', torch.ones_like(self.weight.data, dtype=torch.bool))

    def weight_quantization(self, weight) -> torch.Tensor:
        # Apply quantization to weight
        if self.precision == "full" or weight is None:
            return weight
        elif self.precision == "binary":
            scale = weight.abs().mean()
            weight_out = torch.sign(weight)*scale
        elif isinstance(self.precision, int):
            w_max = weight.max()
            w_min = weight.min()
            weight_out = (w_max - w_min) / (2**self.precision - 1)
            return torch.round((weight - w_min) / scale) * scale + w_min
        else: 
            raise ValueError(f"Unsupported precision type: {self.precision}. Supported types are 'full', 'binary', or a positive integer for quantization bits.")
        # Apply qmask
        if self.qsparsity > 0.0 and hasattr(self, 'qmask'):
            weight_out = weight_out * self.qmask + weight * (~self.qmask)
        print(weight_out)
        return weight_out
        
    def weight_lrd(self, weight1, weight2) -> torch.Tensor:
        if self.rank == "full" or weight1 is None or weight2 is None:
            return weight1
        elif isinstance(self.rank, int):
            return weight1[:, :self.rank] @ weight2[:self.rank, :]
        else:
            raise ValueError(f"Unsupported rank type: {self.rank}. Supported types are 'full' or a positive integer for low-rank decomposition.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # Skip the layer if out_features is 0
        if self.skip:
            return x

        # Soft compression route
        weight = self.weight
        weight_2 = getattr(self, "weight_2", None)
        weight = self.weight_quantization(weight)
        weight_2 = self.weight_quantization(weight_2)
        weight = self.weight_lrd(weight, weight_2)
        
        return F.linear(x, weight, self.bias)    

    def __str__(self):
        return self.__repr__()
    
__all__ = ["LinearCompressed"]