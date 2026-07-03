import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x):
        # Normalize by max absolute value for stability
        max_x = torch.max(x.abs(), dim=-1, keepdim=True).values
        max_x = torch.clamp(max_x, min=1.0)  # Scaling is useless if max_x < 1
        x = x / max_x

        # Evaluate variance
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # Normalize with variance
        x = x * torch.rsqrt(variance + 1e-5)
    

        # Multiply element-wise with the learned weights
        x = self.weight * x
        return x

__all__ = [
    "RMSNorm"
]