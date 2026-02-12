"""Utility functions for compression configuration across different transformer models."""

from typing import Dict, Any, Optional
from .validators import *

COMPRESSION_REGISTRY = {
    "pruning_ratios": dict(default=0.0, validator=validate_pruning_ratio),
    "pruning_modes": dict(default="structured", validator=validate_pruning_mode),
    "lrd_ranks": dict(default="full", validator=validate_lrd_rank),
    "quantization_bits": dict(default=32, validator=None),
}

def init_compressed_config(
    config_instance,
    indexing: Dict[str, Any],
    compression: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Initialize compression configuration for any transformer model.
    """

    path_template = indexing["path_template"]
    depth = getattr(config_instance, indexing["num_blocks_attr"])

    paths = [
        path_template.format(block_index=i, path=p)
        for i in range(depth)
        for p in indexing["path_list"]
    ]

    compression = compression or {}

    for name, meta in COMPRESSION_REGISTRY.items():
        default = meta["default"]

        # Create default dict
        values = {k: default for k in paths}

        # Override with provided values if available
        if name in compression:
            validator = meta["validator"]
            for k, v in compression[name].items():
                if validator:
                    validator(v)
                values[k] = v

        setattr(config_instance, name, values)
    
__all__ = ["init_compressed_config"]