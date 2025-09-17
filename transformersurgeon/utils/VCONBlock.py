import torch
import torch.nn as nn

class VCONBlock(nn.Module):
    def __init__(self,
                 block_a: nn.Module,
                 block_b: nn.Module = None,
                 ):
        super().__init__()
        self.block_a = block_a
        self.block_b = block_b
        self.beta = None  # To be set externally if needed

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.beta is None or self.beta == 1:
            return self.block_a(input)
        elif self.beta == 0:
            if self.block_b is None:
                raise ValueError("Block B is not defined.")
            return self.block_b(input)
        else:
            if self.block_b is None:
                raise ValueError("Block B is not defined.")
            return self.beta * self.block_a(input) + (1 - self.beta) * self.block_b(input)
        
    def set_beta(self, beta: float):
        if beta < 0:
            raise ValueError("Beta value must be non-negative.")
        if beta > 1:
            raise ValueError("Beta value must be less than or equal to 1.")
        self.beta = beta

    def __repr__(self):
        args = [
                f"    block_a={self.block_a}",
                f"    block_b={self.block_b}",
                f"    beta={self.beta}"
            ]
        return "VCONBlock(\n" + ",\n".join(args) + "\n)"
    
    def __str__(self):
        return self.__repr__()        
            
__all__ = ["VCONBlock"]