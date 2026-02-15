import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        # Convert to float for stability
        x = x.to(torch.float32)
        # Evaluate variance
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Normalize with variance
        x = x * torch.rsqrt(variance + self.eps)
        # Multiply element-wise with the learned weights
        x = self.weight * x.to(dtype)

        return x

__all__ = [
    "RMSNorm"
]