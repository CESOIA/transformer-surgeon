"""
VCONBlock.py

Defines the VCONBlock class for vanishing contribution blocks, enabling smooth transitions between original and compressed modules.
"""

import torch
import torch.nn as nn

class VCONBlock(nn.Module):
    """
    Vanishing Contribution Block (VCONBlock).

    Combines the outputs of the original and compressed modules using an affine combination controlled by beta.
    When beta=1, the output is from the original module (block_a); when beta=0, the output is from the compressed module (block_b).

    Args:
        original_module (nn.Module): The original (uncompressed) module.
        compressed_module (nn.Module): The compressed module.

    Methods:
        forward: Computes the affine combination of original and compressed outputs.
        set_beta: Sets the beta parameter.
    """
    def __init__(self,
                 block_a: nn.Module,
                 block_b: nn.Module = None,
                 ):
        super().__init__()
        self.block_a = block_a
        self.block_b = block_b
        self.beta = None  # To be set externally if needed

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.beta is None or self.beta >= 1:
            return self.block_a(input)
        elif self.beta <= 0:
            if self.block_b is None:
                raise ValueError("Block B is not defined.")
            return self.block_b(input)
        else:
            if self.block_b is None:
                raise ValueError("Block B is not defined.")
            return self.beta * self.block_a(input) + (1 - self.beta) * self.block_b(input)
        
    def set_beta(self, beta: float):
        """
        Sets the beta parameter for the VCON block.
         Args:
             beta (float): The beta value to set (0 <= beta <= 1).
         Raises:
             ValueError: If beta is not in the range [0, 1].
    """
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