"""Utility functions for compression configuration across different transformer models."""

from typing import Dict, Any, Optional
from ..compression import COMPRESSION_REGISTRY

def init_compressed_config(
    config_instance,
    indexing: Dict[str, Any],
    compression_config: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
) -> None:
    """
    Initialize compression configuration for any transformer model.
    This will be stored in "compression_config" attribute of the model's configuration instance.

    Args:
        config_instance: The configuration instance of the model to be compressed.
        indexing: A dictionary containing indexing information for the model's layers.
        compression: A nested dictionary specifying compression settings for each path and compression type.
                    The structure should be:
                    {
                        "path1": {
                            "compression_type1": {
                                "property1": value,
                                "property2": value,
                                ...
                            },
                            "compression_type2": {
                                "property1": value,
                                "property2": value,
                                ...
                            },
                            ...
                        },
                        "path2": {
                            ...
                        },
                        ...
                    }
    """

    # Get path template and depth information from indexing
    path_template = indexing["path_template"]
    depth = getattr(config_instance, indexing["num_blocks_attr"])

    # Build the list of all layer paths based on the indexing information
    paths = [
        path_template.format(block_index=i, path=p)
        for i in range(depth)
        for p in indexing["path_list"]
    ]

    # Explore compression input and set default values for missing properties
    compression_config = compression_config or {}
    # For each path
    for path in paths:
        # Get compression settings for this path, generate it if not specified
        path_config = compression_config.setdefault(path, {})
        # For each compression type
        for cname, properties in COMPRESSION_REGISTRY.items():
            # Get compression settings for this compression type
            comp_config = path_config.setdefault(cname, {})
            # For each compression property
            for pname, meta in properties.items():
                # Check if compression property is specified for this path and compression type
                value = comp_config.setdefault(pname, meta["default"])
                # Validate the value if a validator is provided
                if meta["validator"] is not None:
                    meta["validator"](value)

    # Set the compression configuration in the model's config instance
    setattr(config_instance, "compression_config", compression_config)
    
__all__ = ["init_compressed_config"]