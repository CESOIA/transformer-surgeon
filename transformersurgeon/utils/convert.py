import warnings
from ..blocks import TransformerDecoder
from .utils import get_submodule
from collections import defaultdict

def nested_dict():
    return defaultdict(nested_dict)

def convert_for_export(model, verbose=False):
    """
    Convert model components to be compatible with export formats like ONNX.

    Args:
        model: The transformer model to be converted.
        indexing: Model-specific indexing dictionary.

    Returns:
        The converted model.
    """

    # Get indexing and config for the whole model (e.g., vision + text)
    whole_indexing = model.indexing
    whole_config = model.config.to_dict()

    # Convert each model component separately
    new_models = {}
    for name, indexing in whole_indexing.items():
        if verbose:
            print(f"Converting {name} component for export...")

        # Check if conversion is supported
        if 'structure' not in indexing:
            warnings.warn(f"Structure not specified for model with name '{name}'. Skipping conversion.")
            new_models[name] = None
            continue

        # Get configuration of the specific model component
        config = whole_config[indexing['config_attr']]

        # Instantiate new model structure based on indexing
        num_blocks = config[indexing['num_blocks_attr']]

        if indexing['structure'] == 'transformer_decoder':
            new_model = _instantiate_decoder_block(config, indexing, num_blocks)
            path_template = "blocks.{i}.{path}"
        else:
            raise NotImplementedError(f"Conversion for structure '{indexing['structure']}' is not implemented.")
        
        # Import parameters from the original model to the new model
        for i in range(num_blocks):
            for old_path, new_path in zip(indexing['path_list'], indexing['layer_matching']):
                old_path = indexing['path_template'].format(block_index=i, path=old_path)
                new_path = path_template.format(i=i, path=new_path)
                if verbose:
                    print("Transfering parameters:", old_path, "->", new_path)
                old_layer = get_submodule(model, old_path)
                new_layer = get_submodule(new_model, new_path)
                new_layer.load_state_dict(old_layer.state_dict())

        # Handle extra layers if any
        if 'extra_layers' in indexing:
            for old_path, new_path in zip(indexing['extra_layers'], indexing['extra_layers_matching']):
                if verbose:
                    print("Transfering extra layer parameters:", old_path, "->", new_path)
                old_layer = get_submodule(model, old_path)
                new_layer = get_submodule(new_model, new_path)
                new_layer.load_state_dict(old_layer.state_dict())

        new_models[name] = new_model

    return new_models
        
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
        bias_required = nested_dict()
        for j, (path, matched) in enumerate(zip(indexing['path_list'], indexing['layer_matching'])):
            full_path_old = indexing['path_template'].format(block_index=i, path=path)

            if matched in ['norm_in', 'norm_out']:
                continue  # Skip normalization layers

            matched_block, matched_layer = matched.split('.')
            
            # Collect compression config parameters
            compression_config[matched_block][matched_layer]['lrd_rank'] = config['lrd_ranks'][full_path_old]
            compression_config[matched_block][matched_layer]['pruning_mode'] = config['pruning_modes'][full_path_old]
            compression_config[matched_block][matched_layer]['pruning_ratio'] = config['pruning_ratios'][full_path_old]

            # Collect bias requirement
            bias_required[matched_block][matched_layer] = indexing['bias_required'][j]

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
            "bias_required": bias_required,
        }

        blocks_config.append(block_config)

    # Define extra layers configuration
    extra_layers_config = {}
    if 'extra_layers_matching' in indexing:
        for extra_layer in indexing['extra_layers_matching']:
            if extra_layer == "norm":
                extra_layers_config["norm"] = {
                    "embed_dim": config[indexing['embed_dim_attr']]
                }

    # Instantiate the TransformerDecoder with the blocks configuration
    decoder = TransformerDecoder(blocks_config, extra_layers_config)
    return decoder