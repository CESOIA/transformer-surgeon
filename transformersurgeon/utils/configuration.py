"""Utility functions for compression configuration across different transformer models."""

from typing import Dict, List, Any, Optional, Union
from transformers.utils import logging

logger = logging.get_logger(__name__)

def validate_pruning_ratio(ratio: float, name: str = "pruning_ratio") -> None:
    """Validate that pruning ratio is between 0.0 and 1.0."""
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, but got {ratio}.")

def expand_scalar_to_list(config_dict: Dict[str, Any], total_blocks: int) -> Dict[str, List]:
    """Convert scalar values in config dict to lists with equal values."""
    if config_dict is not None:
        for key, value in config_dict.items():
            if isinstance(value, (int, bool, float, str)):
                config_dict[key] = [value] * total_blocks
    return config_dict

def apply_general_keys(config_dict: Dict[str, Any], target_dict: Dict[str, List], key_mappings: Dict[str, List[str]]) -> None:
    """Apply general-type keys to substitute specific layer types."""
    if config_dict is not None:
        # Apply individual keys
        target_dict.update({k: v for k, v in config_dict.items() 
                           if k not in key_mappings.keys()})
        
        # Apply group keys (including "all")
        for group_key, target_keys in key_mappings.items():
            if config_dict.get(group_key) is not None:
                for target_key in target_keys:
                    print(f"Info: overwriting config '{target_key}' with general config '{group_key}'")
                    target_dict[target_key] = config_dict[group_key]
        
def calculate_pruned_dimensions(
    pruning_ratio_lists: Dict[str, List[float]], 
    base_dim: int, 
    mlp_ratio: Optional[float] = None, 
    num_heads: Optional[int] = None, 
    intermediate_size: Optional[int] = None
) -> Dict[str, List[int]]:
    """Calculate pruned dimensions from pruning ratios."""
    pruned_dim_lists = {}
    
    for key, ratio_list in pruning_ratio_lists.items():
        if ratio_list is not None and isinstance(ratio_list, list):
            pruned_dim_list = []
            
            for ratio in ratio_list:
                validate_pruning_ratio(ratio, f"pruning ratio for {key}")
                
                # Determine base dimension for this layer type
                if key in ["mlp_up", "mlp_gate"] and mlp_ratio is not None:
                    embed_dim = base_dim * mlp_ratio
                elif key == "mlp_up" and intermediate_size is not None:
                    embed_dim = intermediate_size
                else:
                    embed_dim = base_dim
                
                # Calculate pruned dimension
                pruned_dim = embed_dim - int(embed_dim * ratio)
                
                # Ensure attention heads are divisible by num_heads
                if key in ["sa_qkv", "sa_q", "sa_k", "sa_v"] and num_heads is not None:
                    pruned_dim = (pruned_dim // num_heads) * num_heads
                
                pruned_dim_list.append(pruned_dim)
            
            pruned_dim_lists[key] = pruned_dim_list
    
    return pruned_dim_lists

def init_compression_config(
    config_instance,
    total_blocks: int,
    base_dim: int,
    pruning_ratio_lists: Optional[Dict[str, Any]] = None,
    pruning_ratio_skip_connections: Optional[float] = None,
    lrd_rank_lists: Optional[Dict[str, Any]] = None,
    default_pruning_keys: Optional[List[str]] = None,
    default_lrd_keys: Optional[List[str]] = None,
    pruning_key_mappings: Optional[Dict[str, List[str]]] = None,
    lrd_key_mappings: Optional[Dict[str, List[str]]] = None,
    mlp_ratio: Optional[float] = None,
    num_heads: Optional[int] = None,
    intermediate_size: Optional[int] = None
) -> None:
    """
    Initialize compression configuration for any transformer model.
    
    Args:
        config_instance: The config instance to modify
        total_blocks: Number of transformer blocks
        base_dim: Base embedding/hidden dimension
        pruning_ratio_lists: Dict of pruning ratios per layer type
        pruning_ratio_skip_connections: Pruning ratio for skip connections
        lrd_rank_lists: Dict of LRD ranks per layer type
        default_pruning_keys: List of default keys for pruning
        default_lrd_keys: List of default keys for LRD
        pruning_key_mappings: Dict mapping general keys to specific layer types for pruning
        lrd_key_mappings: Dict mapping general keys to specific layer types for LRD
        mlp_ratio: MLP expansion ratio (for vision models)
        num_heads: Number of attention heads
        intermediate_size: Intermediate size for MLP (for text models)
    """
    
    # Validate and set skip connection pruning ratio
    config_instance.pruning_ratio_skip_connections = pruning_ratio_skip_connections
    if pruning_ratio_skip_connections is not None:
        validate_pruning_ratio(pruning_ratio_skip_connections, "pruning_ratio_skip_connections")
        config_instance.pruned_embed_dim = base_dim - int(base_dim * pruning_ratio_skip_connections)
    else:
        config_instance.pruned_embed_dim = base_dim
    
    # Set attribute name based on model type (for text models, use pruned_hidden_size)
    if hasattr(config_instance, 'hidden_size'):
        config_instance.pruned_hidden_size = config_instance.pruned_embed_dim
    
    # Initialize default pruning ratio lists
    config_instance.pruning_ratio_lists = {
        key: [0.0] * total_blocks for key in (default_pruning_keys or [])
    }
    
    # Process pruning ratio lists
    pruning_ratio_lists = expand_scalar_to_list(pruning_ratio_lists, total_blocks)
    if pruning_ratio_lists is not None:
        apply_general_keys(pruning_ratio_lists, config_instance.pruning_ratio_lists, pruning_key_mappings or {})
    
    # Initialize default LRD rank lists
    config_instance.lrd_rank_lists = {
        key: ["full"] * total_blocks for key in (default_lrd_keys or [])
    }
    
    # Process LRD rank lists
    lrd_rank_lists = expand_scalar_to_list(lrd_rank_lists, total_blocks)
    if lrd_rank_lists is not None:
        apply_general_keys(lrd_rank_lists, config_instance.lrd_rank_lists, lrd_key_mappings or {})
    
    # Calculate pruned dimensions
    config_instance.pruned_dim_lists = calculate_pruned_dimensions(
        config_instance.pruning_ratio_lists,
        base_dim,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        intermediate_size=intermediate_size
    )

__all__ = [
    "validate_pruning_ratio",
    "expand_scalar_to_list",
    "apply_general_keys",
    "calculate_pruned_dimensions",
    "init_compression_config"
]