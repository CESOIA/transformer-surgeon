# Simple blocks implementation for transformers
# Multi Layer Perceptron (MLP) blocks
import torch
from . import LinearCompressed

# Activation function mapping
activation_map = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "silu": torch.nn.SiLU,
}

class MLP(torch.nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            bias_required=None,
            activation='gelu',
            compression_config=None
            ):
        """
        Standard MLP block with two linear layers and an activation in between.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            bias (bool): Whether to use bias in linear layers.
            activation (str): Activation function to use ('gelu', 'relu', 'silu').
            compression_config (dict, optional): Compression configuration for the linear layers.
        """
        super().__init__()

        # Setup compression configuration
        if compression_config is None:
            compression_config = {
                "up_proj": {"lrd_rank": "full"},
                "down_proj": {"lrd_rank": "full"}
            }
        
        up_proj_lrd_rank = compression_config["up_proj"]["lrd_rank"]
        down_proj_lrd_rank = compression_config["down_proj"]["lrd_rank"]

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "up_proj": False,
                "down_proj": False
            }

        self.up_proj = LinearCompressed(
            embed_dim,
            hidden_dim,
            bias=bias_required["up_proj"],
            lrd_rank=up_proj_lrd_rank)
        
        self.activation = activation_map[activation]()

        self.down_proj = LinearCompressed(
            hidden_dim,
            embed_dim,
            bias=bias_required["down_proj"],
            lrd_rank=down_proj_lrd_rank)

    def forward(self, x):
        """
        Forward pass of the MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, output_dim).
        """
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x
    
class MLPGated(torch.nn.Module): # Qwen-style gated MLP
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            bias_required=None,
            activation='gelu',
            compression_config=None
            ):
        """
        Gated MLP block with two linear layers for gating and one for output projection.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            bias (bool): Whether to use bias in linear layers.
            activation (callable): Activation function class to use.
            compression_config (dict, optional): Compression configuration for the linear layers.
        """
        super().__init__()

        # Setup compression configuration
        if compression_config is None:
            compression_config = {
                "up_proj": {"lrd_rank": "full"},
                "gate_proj": {"lrd_rank": "full"},
                "down_proj": {"lrd_rank": "full"}
            }

        up_proj_lrd_rank = compression_config["up_proj"]["lrd_rank"]
        gate_proj_lrd_rank = compression_config["gate_proj"]["lrd_rank"]
        down_proj_lrd_rank = compression_config["down_proj"]["lrd_rank"]

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "up_proj": False,
                "gate_proj": False,
                "down_proj": False
            }

        self.up_proj = LinearCompressed(
            embed_dim,
            hidden_dim,
            bias=bias_required["up_proj"],
            lrd_rank=up_proj_lrd_rank)
        
        self.gate_proj = LinearCompressed(
            embed_dim,
            hidden_dim,
            bias=bias_required["gate_proj"],
            lrd_rank=gate_proj_lrd_rank)
        
        self.activation = activation_map[activation]()

        self.down_proj = LinearCompressed(
            hidden_dim,
            embed_dim,
            bias=bias_required["down_proj"],
            lrd_rank=down_proj_lrd_rank)

    def forward(self, x):
        """
        Forward pass of the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, output_dim).
        """
        x_up = self.up_proj(x)
        x_gate = self.gate_proj(x)
        x = self.activation(x_gate) * x_up
        x = self.down_proj(x)
        return x
    
__all__ = [
    "MLP",
    "MLPGated",
]
