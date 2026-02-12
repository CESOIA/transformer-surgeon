from typing import Dict, List, Any, Optional, Union

def validate_pruning_ratio(ratio: float) -> None:
    """
    Validate that pruning ratio is between 0.0 and 1.0.
    Args:
        ratio (float): The pruning ratio to validate.
    """
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Pruning ratio must be between 0.0 and 1.0, but got {ratio}.")
    
def validate_pruning_mode(mode: str) -> None:
    """
    Validate that pruning mode is either 'structured' or 'unstructured'.
    Args:
        mode (str): The pruning mode to validate.
    """
    valid_modes = ["structured", "unstructured"]
    if mode is not None and mode not in valid_modes:
        raise ValueError(f"Pruning mode must be one of {valid_modes}, but got '{mode}'.")
    
def validate_lrd_rank(rank: Union[int, str]) -> None:
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

__all__ = [
    "validate_pruning_ratio",
    "validate_pruning_mode",
    "validate_lrd_rank"
]