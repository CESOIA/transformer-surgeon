# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
import warnings
import torch
from transformers import PretrainedConfig
from ..blocks import LinearCompressed, EmbeddingCompressed, Conv2dCompressed, Conv3dCompressed
from typing import Dict, Any
import math
from .utils import flatten_index_paths

def replace_layers_upon_init(
        model: torch.nn.Module, # model to modify (e.g., vision encoder, language decoder)
        indexing: Dict[str, Any], # indexing dictionary for the specific model
        config: PretrainedConfig, # block specific configuration (i.e., vision encoder, language decoder)
        ):
    """
    Replace Linear layers with LinearCompressed layers upon model initialization based on the provided indexing and configuration.

    Args:
        model (torch.nn.Module): The model to modify.
        indexing (Dict[str, Any]): Model-specific indexing dictionary.
        config (PretrainedConfig): Block-specific configuration (e.g., vision encoder, language decoder).
        
    Returns:
        None: The model is modified in place.
    """
    if indexing["config_attr"] != "":
        config_dict = getattr(config, indexing["config_attr"]).to_dict()
    else:
        config_dict = config.to_dict()

    # Extract layer path information
    path_template = indexing['path_template']
    path_list = flatten_index_paths(indexing['path_list'])
    blocks_num = config_dict.get(indexing['num_blocks_attr'], None)
    if blocks_num is None:
        raise ValueError(f"Configuration does not contain the attribute '{indexing['num_blocks_attr']}' required to determine the number of blocks.")
    
    # Navigate through the model to replace Linear layers with LinearCompressed
    for block_idx in range(blocks_num): # for each transformer block
        for path in path_list: # for each layer in the block

            full_path = path_template.format(block_index=block_idx, path=path)
            split_path = full_path.split('.')

            # Reconstruct the module hierarchy to access the target layer
            parent_module = model
            for path_piece in split_path[:-1]:
                parent_module = getattr(parent_module, path_piece, None)
                if parent_module is None:
                    raise AttributeError(f"Module '{path_piece}' not found in the model while accessing '{full_path}'")
        
            # Get the target module
            module_name = split_path[-1]
            old_module = getattr(parent_module, module_name, None)

            # Substitute Linear modules with LinearCompressed
            if type(old_module) is torch.nn.Linear:
                
                # Get the module parameters
                # WIP logic for determining features size from pruning config
                in_features = old_module.in_features
                out_features = old_module.out_features
            
                # Get lrd rank from config if available
                compression_config = config_dict.get('compression_config', {})
                compression_config = compression_config.get(full_path, {})
                lrd_config = compression_config.get("lrd", {})
                rank = lrd_config.get("rank", None)

                new_module = LinearCompressed(
                    in_features=in_features,
                    out_features=out_features,
                    bias=old_module.bias is not None,
                    device=old_module.weight.device,
                    dtype=old_module.weight.dtype,
                    rank=rank,
                    )

                setattr(parent_module, module_name, new_module)

    # Substitute the singleton preprocessing/final_layer/preprocessing_conv
    # modules (not part of path_list's per-block loop -- each is a single,
    # already-resolved path).
    for full_path, type_map, get_config in (
        (indexing.get("preprocessing"), {torch.nn.Embedding: EmbeddingCompressed}, _embedding_new_module_kwargs),
        (indexing.get("final_layer"), {torch.nn.Linear: LinearCompressed}, _linear_new_module_kwargs),
        (indexing.get("preprocessing_conv"), _CONV_TYPE_MAP, _conv_new_module_kwargs),
    ):
        if not full_path:
            continue
        _replace_singleton_layer(model, full_path, type_map, get_config, config_dict)


# Conv patch-embed preprocessing: either Conv2d (e.g. ViT) or Conv3d (e.g.
# Qwen-VL's temporal patches) -- the substituted class depends on which the
# original module actually is.
_CONV_TYPE_MAP = {
    torch.nn.Conv2d: Conv2dCompressed,
    torch.nn.Conv3d: Conv3dCompressed,
}


def _conv_new_module_kwargs(old_module, compression_config):
    return dict(
        in_channels=old_module.in_channels,
        out_channels=old_module.out_channels,
        kernel_size=old_module.kernel_size,
        stride=old_module.stride,
        padding=old_module.padding,
        dilation=old_module.dilation,
        groups=old_module.groups,
        bias=old_module.bias is not None,
        device=old_module.weight.device,
        dtype=old_module.weight.dtype,
    )


def _linear_new_module_kwargs(old_module, compression_config):
    lrd_config = compression_config.get("lrd", {})
    return dict(
        in_features=old_module.in_features,
        out_features=old_module.out_features,
        bias=old_module.bias is not None,
        device=old_module.weight.device,
        dtype=old_module.weight.dtype,
        rank=lrd_config.get("rank", None),
    )


def _embedding_new_module_kwargs(old_module, compression_config):
    lrd_config = compression_config.get("lrd", {})
    return dict(
        num_embeddings=old_module.num_embeddings,
        embedding_dim=old_module.embedding_dim,
        device=old_module.weight.device,
        dtype=old_module.weight.dtype,
        rank=lrd_config.get("rank", None),
    )


def _replace_singleton_layer(model, full_path, type_map, get_config, config_dict):
    """Substitute the module at ``full_path`` with its compressed counterpart.

    Handles ``preprocessing``/``final_layer``/``preprocessing_conv`` indexing
    entries: a single concrete module path (no ``path_template``/block loop).
    ``type_map`` maps the original module's exact type to the compressed
    class to substitute (more than one entry when several original types are
    acceptable, e.g. Conv2d vs Conv3d patch-embed). Silently skips (with a
    warning) modules whose type isn't in ``type_map`` (e.g. a composite
    BERT-style embeddings module), since those aren't supported.
    """
    split_path = full_path.split('.')
    parent_module = model
    for path_piece in split_path[:-1]:
        parent_module = getattr(parent_module, path_piece, None)
        if parent_module is None:
            raise AttributeError(f"Module '{path_piece}' not found in the model while accessing '{full_path}'")

    module_name = split_path[-1]
    old_module = getattr(parent_module, module_name, None)

    compressed_cls = type_map.get(type(old_module))
    if compressed_cls is None:
        if old_module is not None:
            expected_names = ", ".join(t.__name__ for t in type_map)
            warnings.warn(
                f"Skipping compression substitution at '{full_path}': expected "
                f"one of ({expected_names}), got {type(old_module).__name__}."
            )
        return

    compression_config = config_dict.get('compression_config', {})
    compression_config = compression_config.get(full_path, {})

    # No weight copy here: replacement happens inside __init__, before
    # from_pretrained's state-dict loading runs -- it only needs to match the
    # target class/shape so the subsequent load (by parameter name) succeeds,
    # mirroring the path_list Linear substitution above.
    new_module = compressed_cls(**get_config(old_module, compression_config))
    setattr(parent_module, module_name, new_module)

__all__ = ["replace_layers_upon_init"]