import warnings
import copy

import torch
import torch.nn as nn

from ..blocks import TransformerDecoder
from ..blocks.encoder import TransformerEncoder
from ..blocks.indexing import CUSTOM_DECODER_INDEXING, CUSTOM_ENCODER_INDEXING
from ..blocks.config import CustomDecoderConfigCompress, CustomEncoderConfigCompress
from .utils import get_submodule, flatten_index_paths


_QUANT_BUFFER_SUFFIXES = (
    "_act_quant_scale", "_act_quant_zero_point",
    "_act_out_quant_scale", "_act_out_quant_zero_point",
    "_torchao_scale",
)
_QUANT_PLAIN_ATTRS = (
    "_torchao_precision", "_torchao_per_channel", "_soft_quant_precision",
    "_act_quant_precision", "_act_quant_method", "_act_quant_scheme",
    "_act_out_quant_precision", "_act_out_quant_method", "_act_out_quant_scheme",
)


def _extract_aqt_scale(w: torch.Tensor) -> torch.Tensor | None:
    """Extract the quantization scale from any torchao quantized weight tensor.

    Tries multiple attribute paths to handle both the old AffineQuantizedTensor
    layout (tensor_impl.scale) and new-style subclasses (direct .scale attribute).
    """
    if hasattr(w, 'tensor_impl') and hasattr(w.tensor_impl, 'scale'):
        return w.tensor_impl.scale.detach().clone()
    scale = getattr(w, 'scale', None)
    if isinstance(scale, torch.Tensor):
        return scale.detach().clone()
    return None


def _load_state_dict_dequantized(old_layer: nn.Module, new_layer: nn.Module) -> None:
    """Copy old_layer's state dict into new_layer, dequantizing quantized tensors first.

    Handles dynamic quantization buffers (_act_quant_scale, _act_quant_zero_point,
    _torchao_scale) that Quantizer.apply() registers on submodules: these appear in
    old_layer's state_dict but not in a freshly-created new_layer, so they are
    pre-registered before load_state_dict is called. Non-tensor quant attributes
    (_torchao_precision, _act_quant_precision, etc.) are copied explicitly afterwards.
    """
    # Build a plain (float/int) state dict. Use duck-typing: any torch.Tensor subclass
    # with .dequantize() is treated as quantized (handles both old AffineQuantizedTensor
    # and new-style IntxWeightOnly tensors from IntxWeightOnlyConfig).
    plain_state = {}
    for key, value in old_layer.state_dict().items():
        if (
            isinstance(value, torch.Tensor)
            and type(value) is not torch.Tensor
            and hasattr(value, 'dequantize')
        ):
            plain_state[key] = value.dequantize()
        else:
            plain_state[key] = value

    # Pre-register dynamic quantization buffers on the matching new_layer submodules
    # so that load_state_dict below can fill them in.
    new_modules = dict(new_layer.named_modules())
    for key, tensor in plain_state.items():
        if any(key.endswith(s) for s in _QUANT_BUFFER_SUFFIXES):
            parent_path, buf_name = key.rsplit(".", 1) if "." in key else ("", key)
            parent_mod = new_modules.get(parent_path)
            if parent_mod is not None and buf_name not in parent_mod._buffers:
                parent_mod.register_buffer(buf_name, torch.empty_like(tensor))

    new_layer.load_state_dict(plain_state)

    # Propagate non-tensor compression metadata (plain attributes not in state_dict).
    for name, old_submod in old_layer.named_modules():
        if not isinstance(old_submod, nn.Linear):
            continue
        new_submod = new_modules.get(name)
        if new_submod is None:
            continue

        for attr in _QUANT_PLAIN_ATTRS:
            if hasattr(old_submod, attr):
                setattr(new_submod, attr, getattr(old_submod, attr))

        # Fallback: if the weight is still a live quantized tensor and the scale was
        # not already stashed into the state dict, extract and register it now.
        w_old = old_submod.weight
        if (
            "_torchao_scale" not in new_submod._buffers
            and isinstance(w_old, torch.Tensor)
            and type(w_old) is not torch.Tensor
        ):
            scale = _extract_aqt_scale(w_old)
            if scale is not None:
                new_submod.register_buffer("_torchao_scale", scale)

        # Derive _torchao_per_channel from the scale buffer (authoritative source).
        if "_torchao_scale" in new_submod._buffers:
            new_submod._torchao_per_channel = new_submod._torchao_scale.numel() > 1


def _flatten_layer_matching(indexing, source_paths):
    """Normalize layer_matching to a flat list aligned with source_paths."""
    layer_matching = indexing.get("layer_matching", source_paths)
    if isinstance(layer_matching, dict):
        flattened = []
        for _, mapped_paths in layer_matching.items():
            if isinstance(mapped_paths, str):
                flattened.append(mapped_paths)
                continue
            if not isinstance(mapped_paths, (list, tuple)):
                raise TypeError(
                    "Indexing 'layer_matching' dict values must be a string or list/tuple of strings, "
                    f"got {type(mapped_paths).__name__}."
                )
            for mapped_path in mapped_paths:
                flattened.append(str(mapped_path))
        layer_matching = flattened
    elif isinstance(layer_matching, (list, tuple)):
        layer_matching = list(layer_matching)
    else:
        raise TypeError(
            "Indexing 'layer_matching' must be a list/tuple or grouped dict, "
            f"got {type(layer_matching).__name__}."
        )

    if len(layer_matching) != len(source_paths):
        raise ValueError(
            "Indexing 'layer_matching' length mismatch. "
            f"Expected {len(source_paths)} entries from path_list, got {len(layer_matching)}."
        )
    return layer_matching


def _build_bias_required_config(layer_matching, source_bias):
    """Build converted bias map for attention/MLP layers."""
    bias_required_config = {
        "attn": {},
        "mlp": {},
    }

    if isinstance(source_bias, dict):
        bias_by_path = source_bias
    elif isinstance(source_bias, (list, tuple)):
        bias_by_path = {
            mapped_path: source_bias[j]
            for j, mapped_path in enumerate(layer_matching)
            if j < len(source_bias)
        }
    else:
        raise TypeError(
            "Indexing 'bias_required' must be a list/tuple or dict, "
            f"got {type(source_bias).__name__}."
        )

    for mapped_path, is_required in bias_by_path.items():
        if mapped_path.startswith("attn."):
            bias_required_config["attn"][mapped_path.split(".", 1)[1]] = bool(is_required)
        elif mapped_path.startswith("mlp."):
            bias_required_config["mlp"][mapped_path.split(".", 1)[1]] = bool(is_required)

    return bias_required_config


def _build_converted_compression_config(indexing, source_config_obj, num_blocks):
    """
    Remap source compression_config keys from original HF paths to converted custom graph paths.
    """
    source_paths = flatten_index_paths(indexing['path_list'])
    layer_matching = _flatten_layer_matching(indexing, source_paths)
    source_cfg = getattr(source_config_obj, 'compression_config', {}) or {}

    converted_cfg = {}
    for i in range(num_blocks):
        for old_path, new_path in zip(source_paths, layer_matching):
            old_full = indexing['path_template'].format(block_index=i, path=old_path)
            new_full = f"blocks.{i}.{new_path}"
            converted_cfg[new_full] = copy.deepcopy(source_cfg.get(old_full, {}))

    return converted_cfg


def _build_converted_decoder_indexing(indexing):
    """Build converted indexing compatible with converted decoder config/model."""
    converted_indexing = copy.deepcopy(CUSTOM_DECODER_INDEXING)
    source_paths = flatten_index_paths(indexing["path_list"])
    layer_matching = _flatten_layer_matching(indexing, source_paths)
    converted_indexing["decoder"]["path_list"] = [
        path
        for path in layer_matching
        if path.startswith("attn.") or path.startswith("mlp.")
    ]
    return converted_indexing


def _build_converted_encoder_indexing(indexing):
    """Build converted indexing compatible with converted encoder config/model."""
    converted_indexing = copy.deepcopy(CUSTOM_ENCODER_INDEXING)
    source_paths = flatten_index_paths(indexing["path_list"])
    layer_matching = _flatten_layer_matching(indexing, source_paths)
    converted_indexing["encoder"]["path_list"] = [
        path
        for path in layer_matching
        if path.startswith("attn.") or path.startswith("mlp.")
    ]
    return converted_indexing

def convert_for_export(model, options=None, verbose=False):
    """
    Convert model components to be compatible with export formats like ONNX.

    Args:
        model: The transformer model to be converted.
        indexing: Model-specific indexing dictionary.

    Returns:
        The converted model.
    """

    options = options or {}

    # Get indexing and config for the whole model (e.g., vision + text)
    whole_indexing = model.indexing
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
            source_config_obj = whole_config_obj
        else:
            source_config_obj = getattr(whole_config_obj, indexing['config_attr'])

        # Instantiate new model structure based on indexing
        num_blocks = getattr(source_config_obj, indexing['num_blocks_attr'])

        if indexing['structure'] == 'transformer_decoder':
            use_sdpa = options.get('use_sdpa', False)
            max_cache_len = options.get('max_cache_len', 2048)
            cache_impl = options.get('cache_impl', 'mutable')

            converted_indexing = _build_converted_decoder_indexing(indexing)
            converted_compression_config = _build_converted_compression_config(
                indexing,
                source_config_obj,
                num_blocks,
            )
            source_paths = flatten_index_paths(indexing["path_list"])
            layer_matching = _flatten_layer_matching(indexing, source_paths)
            source_bias = indexing.get("bias_required", [])
            bias_required_config = _build_bias_required_config(layer_matching, source_bias)

            converted_config = CustomDecoderConfigCompress.from_source_config(
                source_config_obj=source_config_obj,
                source_indexing=indexing,
                converted_indexing=converted_indexing["decoder"],
                compression_config=converted_compression_config,
                bias_required=bias_required_config,
                use_sdpa=use_sdpa,
                max_cache_len=max_cache_len,
                cache_impl=cache_impl,
            )

            new_model = TransformerDecoder(config=converted_config)
            path_template = "blocks.{i}.{path}"
        elif indexing['structure'] == 'transformer_encoder':
            use_sdpa = options.get('use_sdpa', False)

            converted_indexing = _build_converted_encoder_indexing(indexing)
            converted_compression_config = _build_converted_compression_config(
                indexing,
                source_config_obj,
                num_blocks,
            )
            source_paths = flatten_index_paths(indexing["path_list"])
            layer_matching = _flatten_layer_matching(indexing, source_paths)
            source_bias = indexing.get("bias_required", [])
            bias_required_config = _build_bias_required_config(layer_matching, source_bias)

            converted_config = CustomEncoderConfigCompress.from_source_config(
                source_config_obj=source_config_obj,
                source_indexing=indexing,
                converted_indexing=converted_indexing["encoder"],
                compression_config=converted_compression_config,
                bias_required=bias_required_config,
                use_sdpa=use_sdpa,
                use_final_norm=indexing.get("use_final_norm", True),
                position_embedding_type=indexing.get("position_embedding_type", "none"),
            )

            new_model = TransformerEncoder(config=converted_config)
            path_template = "blocks.{i}.{path}"
        else:
            raise NotImplementedError(f"Conversion for structure '{indexing['structure']}' is not implemented.")
        
        # Import parameters from the original model to the new model
        source_paths = flatten_index_paths(indexing['path_list'])
        layer_matching = _flatten_layer_matching(indexing, source_paths)
        for i in range(num_blocks):
            for old_path, new_path in zip(source_paths, layer_matching):
                old_path = indexing['path_template'].format(block_index=i, path=old_path)
                new_path = path_template.format(i=i, path=new_path)
                old_layer = get_submodule(model, old_path)
                new_layer = get_submodule(new_model, new_path)
                _load_state_dict_dequantized(old_layer, new_layer)
                if verbose:
                    print("Transfering parameters:", old_path, "->", new_path)
                    for pname, param in old_layer.named_parameters():
                        print(f"  {pname}: {param.shape}, dtype={param.dtype}, device={param.device}")
                    print("   -->")
                    for pname, param in new_layer.named_parameters():
                        print(f"  {pname}: {param.shape}, dtype={param.dtype}, device={param.device}")

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
                _load_state_dict_dequantized(old_layer, new_layer)

        # Attach converted indexing so manager can be used directly on converted graph.
        new_model.indexing = converted_indexing

        new_models[name] = new_model

    return new_models