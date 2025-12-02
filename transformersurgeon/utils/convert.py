import torch
from ..blocks import TransformerDecoder
from .utils import get_submodule
from collections import defaultdict

def nested_dict():
    return defaultdict(nested_dict)

def convert_for_export(model, indexing):
    """
    Convert model components to be compatible with export formats like ONNX.

    Args:
        model: The transformer model to be converted.
        indexing: Model-specific indexing dictionary.

    Returns:
        The converted model.
    """

    config = model.config.to_dict()
    num_blocks = config[indexing['num_blocks_attr'])]

    if indexing['structure'] == 'transformer_decoder':
        new_model = _instantiate_decoder_block(config, indexing, num_blocks)
        path_template = "blocks.{i}.{path}"
    else:
        raise NotImplementedError(f"Conversion for structure '{indexing['structure']}' is not implemented.")
    
    # Import parameters from the original model to the new model

    for i in range(num_blocks):
        for old_path, new_path in zip(indexing['path_template'], indexing['layer_matching']):
            old_path = indexing['path_template'].format(block_index=i, path=old_path)
            new_path = path_template.format(i=i, path=new_path)
            if new_path.endswith('.norm_in') or new_path.endswith('.norm_out'):
                continue  # Skip normalization layers
            old_layer = get_submodule(model, old_path)
            new_layer = get_submodule(new_model, new_path)
            # WIP: copy parameters from old_layer to new_layer

    return new_model
        
def _instantiate_decoder_block(config, indexing, blocks_num):
    """
    Instantiate a TransformerDecoder block based on the provided indexing and hf configuration.
    TransformerDecoder configuration is built as a list of independent block configurations, each defined as a nested dictionary.

    Args:
        config_dict: Configuration dictionary for the model.
        indexing: Model-specific indexing dictionary.
        blocks_num: Number of blocks in the transformer.
    """
    # Generate block configuration list from the hf configuration
    blocks_config = []
    for i in range(blocks_num):

        # Define compression configuration as a nested dictionary
        compression_config = nested_dict()
        for path, matched in zip(indexing['path_list'], indexing['layer_matching']):
            full_path_old = indexing['path_template'].format(block_index=i, path=path)

            if matched in ['norm_in', 'norm_out']:
                continue  # Skip normalization layers

            matched_block, matched_layer = matched.split('.')
            
            compression_config[matched_block][matched_layer]['lrd_rank'] = config['lrd_ranks'][full_path_old]
            compression_config[matched_block][matched_layer]['pruning_mode'] = config['pruning_modes'][full_path_old]
            compression_config[matched_block][matched_layer]['pruning_ratio'] = config['pruning_ratios'][full_path_old]

        # Define block configuration
        embed_dim = config[indexing['embed_dim_attr']]
        num_heads = config[indexing['num_heads_attr']]
        mlp_hidden_dim = config[indexing['mlp_hidden_dim_attr']]
        mlp_activation = config[indexing['mlp_activation_attr']]
        kv_num_heads = None if 'kv_num_heads_attr' not in indexing else config[indexing['kv_num_heads_attr']]
        mha_type = indexing['attn_type']
        mlp_type = indexing["mlp_type"]
        norm_type = indexing["norm_type"]

        block_config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_activation": mlp_activation,
            "kv_num_heads": kv_num_heads,
            "mha_type": mha_type,
            "mlp_type": mlp_type,
            "norm_type": norm_type,            
            "compression_config": compression_config,
        }

        blocks_config.append(block_config)

    # Instantiate the TransformerDecoder with the blocks configuration
    decoder = TransformerDecoder(blocks_config)
    return decoder