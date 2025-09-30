"""Utility functions for compression configuration across different transformer models."""

from typing import Dict, List, Any, Optional, Union
from transformers.utils import logging

logger = logging.get_logger(__name__)

def _validate_pruning_ratio(ratio: float) -> None:
    """
    Validate that pruning ratio is between 0.0 and 1.0.
    Args:
        ratio (float): The pruning ratio to validate.
    """
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Pruning ratio must be between 0.0 and 1.0, but got {ratio}.")

def _expand_scalar_to_list(config_dict: Dict[str, Any], total_blocks: int) -> Dict[str, List]:
    """
    Convert scalar values in config dict to lists with equal values.
    Args:
        config_dict (Dict[str, Any]): The configuration dictionary.
        total_blocks (int): The total number of transformer blocks.
    Returns:
        Dict[str, List]: The updated configuration dictionary with lists.
    """
    if config_dict is not None:
        for key, value in config_dict.items():
            if isinstance(value, (int, bool, float, str)):
                config_dict[key] = [value] * total_blocks
    return config_dict

def _apply_general_keys(config_dict: Dict[str, Any], target_dict: Dict[str, List], key_mappings: Dict[str, List[str]]) -> None:
    """
    Apply general-type keys to substitute specific layer types.
    Args:
        config_dict (Dict[str, Any]): The input configuration dictionary with potential general keys.
        target_dict (Dict[str, List]): The target configuration dictionary to update.
        key_mappings (Dict[str, List[str]]): Mapping from general keys to specific layer types.
    """
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
    indexing: Dict[str, Any],
    pruning_ratios: Optional[Dict[str, Any]] = None,
    lrd_ranks: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Initialize compression configuration for any transformer model.
    
    Args:
        config_instance: The configuration instance to initialize (e.g., model.config).
        indexing (Dict[str, Any]): Model-specific indexing dictionary.
        pruning_ratio_dict (Optional[Dict[str, Any]]): Dictionary specifying pruning ratios.
        lrd_rank_dict (Optional[Dict[str, Any]]): Dictionary specifying LRD ranks
    """

    path_template = indexing["path_template"]
    depth = getattr(config_instance, indexing["num_blocks_attr"])
    # Generate default config values by building all the paths in a dictionary
    config_instance.pruning_ratios = {path_template.format(block_index=id, path=path): 0.0 for id in range(depth) for path in indexing["path_list"]}
    config_instance.lrd_ranks = {path_template.format(block_index=id, path=path): "full" for id in range(depth) for path in indexing["path_list"]}
    
    # Get values from arguments
    if pruning_ratios is not None:
        for key, value in pruning_ratios.items():
            _validate_pruning_ratio(value)
            config_instance.pruning_ratios[key] = value
    if lrd_ranks is not None:
        for key, value in lrd_ranks.items():
            config_instance.lrd_ranks[key] = value
    
__all__ = ["init_compression_config"]