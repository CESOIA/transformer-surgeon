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
    
def _validate_pruning_mode(mode: str) -> None:
    """
    Validate that pruning mode is either 'structured' or 'unstructured'.
    Args:
        mode (str): The pruning mode to validate.
    """
    valid_modes = ["structured", "unstructured"]
    if mode is not None and mode not in valid_modes:
        raise ValueError(f"Pruning mode must be one of {valid_modes}, but got '{mode}'.")
    
def _validate_lrd_rank(rank: Union[int, str]) -> None:
    """
    Validate that LRD rank is either 'full' or a positive integer.
    Args:
        rank (Union[int, str]): The LRD rank to validate.
    """
    if rank is not None:
        if isinstance(rank, str):
            if rank != "full":
                raise ValueError(f"LRD rank must be 'full' or a positive integer, but got '{rank}'.")
        elif isinstance(rank, int):
            if rank <= 0:
                raise ValueError(f"LRD rank must be a positive integer, but got {rank}.")
        else:
            raise ValueError(f"LRD rank must be 'full' or a positive integer, but got type {type(rank)}.")

def init_compression_config(
    config_instance,
    indexing: Dict[str, Any],
    pruning_ratios: Optional[Dict[str, Any]] = None,
    pruning_mode: Optional[Dict[str, Any]] = None,
    lrd_ranks: Optional[Dict[str, Any]] = None,
    quantization_bits: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Initialize compression configuration for any transformer model.
    
    Args:
        config_instance: The configuration instance to initialize (e.g., model.config).
        indexing (Dict[str, Any]): Model-specific indexing dictionary.
        pruning_ratio_dict (Optional[Dict[str, Any]]): Dictionary specifying pruning ratios.
        lrd_rank_dict (Optional[Dict[str, Any]]): Dictionary specifying LRD ranks
        quantization_bits_dict (Optional[Dict[str, Any]]): Dictionary specifying quantization bits.
    """

    path_template = indexing["path_template"]
    depth = getattr(config_instance, indexing["num_blocks_attr"])
    # Generate default config values by building all the paths in a dictionary
    config_instance.pruning_ratios = {path_template.format(block_index=id, path=path): 0.0 for id in range(depth) for path in indexing["path_list"]}
    config_instance.pruning_modes = {path_template.format(block_index=id, path=path): "structured" for id in range(depth) for path in indexing["path_list"]}
    config_instance.lrd_ranks = {path_template.format(block_index=id, path=path): "full" for id in range(depth) for path in indexing["path_list"]}
    config_instance.quantization_bits = {path_template.format(block_index=id, path=path): 32 for id in range(depth) for path in indexing["path_list"]}
    # Get values from arguments
    if pruning_ratios is not None:
        for key, value in pruning_ratios.items():
            _validate_pruning_ratio(value)
            config_instance.pruning_ratios[key] = value
    if pruning_mode is not None:
        for key, value in pruning_mode.items():
            _validate_pruning_mode(value)
            config_instance.pruning_modes[key] = value
    if lrd_ranks is not None:
        for key, value in lrd_ranks.items():
            _validate_lrd_rank(value)
            config_instance.lrd_ranks[key] = value
    if quantization_bits is not None:
        for key, value in quantization_bits.items():
            config_instance.quantization_bits[key] = value
    
__all__ = ["init_compression_config"]