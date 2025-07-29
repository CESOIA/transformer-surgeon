import torch
import torch.nn as nn
from typing import Union

# Custom linear layer for low-rank decomposition

# Note on low-rank decomposition:
#   when using low-rank decomposition, concatenate the two decomposed matrices into a single weight tensor.
#   weight = torch.cat((weightA, weightB.T), dim=0)

class LinearLRD(nn.Linear):
    def __init__(self, 
                 in_features, 
                 out_features,
                 bias=False,
                 lrd_rank: Union[int, str] = "full"):
        
        self.lrd_rank = lrd_rank
        if isinstance(lrd_rank, int):
            if lrd_rank < 1:
                raise ValueError("Low-rank decomposition rank must be at least 1.")
            # use "weight" tensor to pack both the two decomposed matrices
            # weightA = weight[:in_features, :]; weightB = weight[in_features:, :]
            in_features = in_features + out_features
            out_features = lrd_rank
        
        if out_features <= 0:
            # If out_features is 0, skip the layer
            self.skip = True
        else:
            self.skip = False
            super().__init__(in_features, out_features, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Skip the layer if out_features is 0
        if self.skip:
            return input
        # Perform normally if lrd_rank is full
        if self.lrd_rank == "full":
            # Full linear layer
            return super().forward(input)
        # Perform with low-rank decomposition
        elif isinstance(self.lrd_rank, int):
            weight1 = self.weight[:self.in_features, :]
            weight2 = self.weight[self.in_features:, :]
            # Compute: x @ weight1 @ weight2.T + bias
            output = input @ weight1 @ weight2.t()
            if self.bias is not None:
                output += self.bias
            return output
        # Manage value errors
        else:
            raise ValueError(f"Unsupported low-rank decomposition value: {self.lrd_rank}")
        
    def __str__(self):
        return super().__str__() + f"(lrd_rank={self.lrd_rank})"
    
__all__ = ["LinearLRD"]