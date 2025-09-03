"""Utility functions for compression configuration across different transformer models."""

from typing import Dict, List, Any, Optional, Union
from transformers.utils import logging

logger = logging.get_logger(__name__)

def _validate_pruning_ratio(ratio: float, name: str = "pruning_ratio") -> None:
    """Validate that pruning ratio is between 0.0 and 1.0."""
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, but got {ratio}.")

def _expand_scalar_to_list(config_dict: Dict[str, Any], total_blocks: int) -> Dict[str, List]:
    """Convert scalar values in config dict to lists with equal values."""
    if config_dict is not None:
        for key, value in config_dict.items():
            if isinstance(value, (int, bool, float, str)):
                config_dict[key] = [value] * total_blocks
    return config_dict

def _apply_general_keys(config_dict: Dict[str, Any], target_dict: Dict[str, List], key_mappings: Dict[str, List[str]]) -> None:
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

def init_compression_config(
    config_instance,
    total_blocks: int,
    indexing: Dict[str, Any],
    pruning_ratio_lists: Optional[Dict[str, Any]] = None,
    pruning_ratio_skip_connections: Optional[float] = None,
    lrd_rank_lists: Optional[Dict[str, Any]] = None,
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

    # Generate keys
    # Define layer type mappings for vision model - separate for pruning and LRD
    default_pruning_keys = [v for v, m in zip(indexing['key_list'], indexing['pruning_supported']) if m]
    default_lrd_keys = [v for v, m in zip(indexing['key_list'], indexing['lrd_supported']) if m]

    # Define key mappings - separate for pruning and LRD
    pruning_key_mappings = {k: [v for v in vals if v in default_pruning_keys] for k, vals in indexing['key_mappings'].items()}
    lrd_key_mappings = {k: [v for v in vals if v in default_lrd_keys] for k, vals in indexing['key_mappings'].items()}
    
    # Validate and set skip connection pruning ratio
    config_instance.pruning_ratio_skip_connections = pruning_ratio_skip_connections
    if pruning_ratio_skip_connections is not None:
        _validate_pruning_ratio(pruning_ratio_skip_connections, "pruning_ratio_skip_connections")
    
    # Initialize default pruning ratio lists
    config_instance.pruning_ratio_lists = {
        key: [0.0] * total_blocks for key in (default_pruning_keys or [])
    }
    
    # Process pruning ratio lists
    pruning_ratio_lists = _expand_scalar_to_list(pruning_ratio_lists, total_blocks)
    if pruning_ratio_lists is not None:
        _apply_general_keys(pruning_ratio_lists, config_instance.pruning_ratio_lists, pruning_key_mappings or {})
    
    # Initialize default LRD rank lists
    config_instance.lrd_rank_lists = {
        key: ["full"] * total_blocks for key in (default_lrd_keys or [])
    }
    
    # Process LRD rank lists
    lrd_rank_lists = _expand_scalar_to_list(lrd_rank_lists, total_blocks)
    if lrd_rank_lists is not None:
        _apply_general_keys(lrd_rank_lists, config_instance.lrd_rank_lists, lrd_key_mappings or {})

__all__ = ["init_compression_config"]