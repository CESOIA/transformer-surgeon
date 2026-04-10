import warnings
from ..blocks import TransformerDecoder
from ..blocks.indexing import CUSTOM_DECODER_INDEXING
from ..blocks.config import build_custom_decoder_config
from .utils import get_submodule
import copy


def _build_converted_compression_config(indexing, source_config_obj, num_blocks):
    """
    Remap source compression_config keys from original HF paths to converted custom graph paths.
    """
    layer_matching = indexing.get('layer_matching', indexing['path_list'])
    source_cfg = getattr(source_config_obj, 'compression_config', {}) or {}

    converted_cfg = {}
    for i in range(num_blocks):
        for old_path, new_path in zip(indexing['path_list'], layer_matching):
            old_full = indexing['path_template'].format(block_index=i, path=old_path)
            new_full = f"blocks.{i}.{new_path}"
            converted_cfg[new_full] = copy.deepcopy(source_cfg.get(old_full, {}))

    return converted_cfg

def convert_for_export(model, options={}, verbose=False):
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
    whole_config_obj = model.config

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
        if indexing['config_attr'] == "":
            config = whole_config
            source_config_obj = whole_config_obj
        else:
            config = whole_config[indexing['config_attr']]
            source_config_obj = getattr(whole_config_obj, indexing['config_attr'])

        # Instantiate new model structure based on indexing
        num_blocks = config[indexing['num_blocks_attr']]

        if indexing['structure'] == 'transformer_decoder':
            use_sdpa = options.get('use_sdpa', False)
            max_cache_len = options.get('max_cache_len', 2048)
            new_model = _instantiate_decoder_block(config, indexing, num_blocks, use_sdpa, max_cache_len)
            path_template = "blocks.{i}.{path}"
        else:
            raise NotImplementedError(f"Conversion for structure '{indexing['structure']}' is not implemented.")
        
        # Import parameters from the original model to the new model
        layer_matching = indexing.get('layer_matching', indexing['path_list'])
        for i in range(num_blocks):
            for old_path, new_path in zip(indexing['path_list'], layer_matching):
                old_path = indexing['path_template'].format(block_index=i, path=old_path)
                new_path = path_template.format(i=i, path=new_path)
                if verbose:
                    print("Transfering parameters:", old_path, "->", new_path)
                old_layer = get_submodule(model, old_path)
                new_layer = get_submodule(new_model, new_path)
                new_layer.load_state_dict(old_layer.state_dict())

        # Handle extra layers if any
        if 'extra_layers' in indexing:
            extra_layers_matching = indexing.get('extra_layers_matching', [])
            if not extra_layers_matching:
                extra_layers_matching = [path.split(".")[-1] for path in indexing['extra_layers']]
            for old_path, new_path in zip(indexing['extra_layers'], extra_layers_matching):
                if verbose:
                    print("Transfering extra layer parameters:", old_path, "->", new_path)
                old_layer = get_submodule(model, old_path)
                new_layer = get_submodule(new_model, new_path)
                new_layer.load_state_dict(old_layer.state_dict())

        # Attach converted config/indexing so manager can be used directly on converted graph.
        converted_indexing = copy.deepcopy(CUSTOM_DECODER_INDEXING)
        converted_indexing["decoder"]["num_blocks_attr"] = indexing["num_blocks_attr"]
        converted_indexing["decoder"]["path_list"] = [
            path
            for path in layer_matching
            if path.startswith("attn.") or path.startswith("mlp.")
        ]
        converted_compression_config = _build_converted_compression_config(
            indexing,
            source_config_obj,
            num_blocks,
        )
        new_model.config = build_custom_decoder_config(
            source_config_obj,
            converted_indexing["decoder"],
            compression_config=converted_compression_config,
        )
        new_model.indexing = converted_indexing

        new_models[name] = new_model

    return new_models
        
def _instantiate_decoder_block(config, indexing, blocks_num, use_sdpa=False, max_cache_len=2048):
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
    layer_matching = indexing.get('layer_matching', indexing['path_list'])
    for i in range(blocks_num):

        # Define compression and bias in strict nested block format.
        compression_config = {
            "attn": {},
            "mlp": {},
        }
        bias_required = {
            "attn": {},
            "mlp": {},
        }
        for j, (path, matched) in enumerate(zip(indexing['path_list'], layer_matching)):
            full_path_old = indexing['path_template'].format(block_index=i, path=path)

            # Collect compression config parameters for compressible layers only.
            if matched.startswith("attn."):
                _, layer_name = matched.split(".", 1)
                old_compression_config = config.get('compression_config', {}).get(full_path_old, {})
                compression_config["attn"][layer_name] = copy.deepcopy(old_compression_config)
            elif matched.startswith("mlp."):
                _, layer_name = matched.split(".", 1)
                old_compression_config = config.get('compression_config', {}).get(full_path_old, {})
                compression_config["mlp"][layer_name] = copy.deepcopy(old_compression_config)

            # Collect bias requirement.
            if 'bias_required' in indexing and j < len(indexing['bias_required']):
                if matched.startswith("attn."):
                    _, layer_name = matched.split(".", 1)
                    bias_required["attn"][layer_name] = indexing['bias_required'][j]
                elif matched.startswith("mlp."):
                    _, layer_name = matched.split(".", 1)
                    bias_required["mlp"][layer_name] = indexing['bias_required'][j]

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
            "use_sdpa": use_sdpa,
            "max_cache_len": max_cache_len,
        }

        blocks_config.append(block_config)

    # Define extra layers configuration
    extra_layers_config = {}
    if 'extra_layers' in indexing:
        extra_layers_matching = indexing.get('extra_layers_matching', [])
        if not extra_layers_matching:
            extra_layers_matching = [path.split(".")[-1] for path in indexing['extra_layers']]
        for extra_layer in extra_layers_matching:
            if extra_layer == "norm":
                extra_layers_config["norm"] = {
                    "embed_dim": config[indexing['embed_dim_attr']]
                }

    # Instantiate the TransformerDecoder with the blocks configuration
    decoder = TransformerDecoder(blocks_config, extra_layers_config)
    return decoder