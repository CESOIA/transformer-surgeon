import torch

# These are simple atomic blocks that are necessary to split the TransformerDecoderBlock into smaller pieces for better modularity and flexibility.

class AtomicSum(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_num = 2

    def forward(self, x1, x2):
        return x1 + x2
    
__all__ = [
    "AtomicSum",
]