# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
import torch
from transformers import PretrainedConfig
from ..layers import LinearCompressed
from typing import Dict, Any

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
    path_list = indexing['path_list']
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
                lrd_rank = config_dict.get('lrd_ranks', {}).get(full_path, "full")

                new_module = LinearCompressed(
                    in_features=in_features,
                    out_features=out_features,
                    bias=old_module.bias is not None,
                    device=old_module.weight.device,
                    dtype=old_module.weight.dtype,
                    lrd_rank=lrd_rank,
                    )
                setattr(parent_module, module_name, new_module)
                
__all__ = ["replace_layers_upon_init"]