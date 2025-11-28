# Simple blocks implementation for transformers
# Multi Layer Perceptron (MLP) blocks
import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=False, activation=torch.nn.GELU):
        super().__init__()
        self.up_proj = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.activation = activation()
        self.down_proj = torch.nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x
    
class MLPGated(torch.nn.Module): # Qwen-style gated MLP
    def __init__(self, input_dim, hidden_dim, output_dim, bias=False, activation=torch.nn.GELU):
        super().__init__()
        self.up_proj = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.gate_proj = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.activation = activation()
        self.down_proj = torch.nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        x_up = self.up_proj(x)
        x_gate = self.gate_proj(x)
        x = self.activation(x_gate) * x_up
        x = self.down_proj(x)
        return x
    
__all__ = [
    "MLP",
    "MLPGated",
]
