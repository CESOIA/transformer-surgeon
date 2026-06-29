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
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype)
        self.linear_V = None # Placeholder for the V factor (in→rank) in low-rank decomposition

        self.init_lrd(rank)

    def init_lrd(self, rank):
        # Set rank and initialize linear_V for low-rank decomposition if needed
        self.rank = "full" if rank is None else rank
        if isinstance(rank, int):
            device = self.weight.device
            dtype = self.weight.dtype

            # Initialize linear_V (V factor: in→rank) as a proper nn.Linear so that
            # torchao's quantize_() can discover and quantize it alongside self.weight.
            if self.linear_V is None or self.linear_V.out_features != rank:
                self.linear_V = nn.Linear(self.in_features, rank, bias=False,
                                          device=device, dtype=dtype)

            # Adjust weight (U·S factor) shape for low-rank decomposition if necessary
            if self.weight.shape[1] != self.rank:

                self.weight = Parameter(
                    torch.empty([self.out_features, self.rank], device=device, dtype=dtype),
                    requires_grad=True)
                torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
    def cancel_lrd(self):
        self.rank = "full"
        device = self.weight.device
        dtype = self.weight.dtype

        # Remove linear_V if it exists
        if self.linear_V is not None:
            self.linear_V = None

        # Revert weight to original shape if it was changed for low-rank decomposition
        if self.weight.shape[1] != self.in_features:
            self.weight = None
            self.weight = Parameter(
                torch.empty([self.out_features, self.in_features], device=device, dtype=dtype),
                requires_grad=True)
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
          
        # Skip logic
        if self.skip:
            return x

        # TODO make reshape optional and only for QNN export, check if view is enough
        # QNN FullyConnected is more robust with rank-2 inputs. Flatten leading
        # dimensions, run linear(s), and then restore the original leading shape.
        restore_shape = None
        if x.dim() > 2:
            restore_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        
        # LRD logic
        if isinstance(self.rank, int):
            x = self.linear_V(x)  # V factor: [*, rank]

        y = F.linear(x, self.weight, self.bias)

        if restore_shape is not None:
            y = y.reshape(*restore_shape, y.shape[-1])

        return y

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"LinearCompressed(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank})" 
    
__all__ = ["LinearCompressed"]