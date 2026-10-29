"""Convert a compressed Hugging Face model into export-ready ``blocks/`` stacks.

``convert_for_export`` reads, for every component of the model dictionary that
declares a ``structure``:

1. the geometry and the atoms the model needs (``structure``, ``attn_type``,
   ``mlp_type``, ``norm_type``, ``rope_type`` and the ``*_attr`` names);
2. the geometry of every already-compressed linear layer (size, rank, bias),
   straight from the source modules;
3. the export implementation options (``attn_impl``, ``cache_impl``, ...).

It then builds the stack named by ``structure`` from the block catalog, copies
the weights layer by layer (``layer_matching`` pairs source and target names)
and calls the attention modules' ``finalize_*`` hooks once. It never decides
anything about compression: it reproduces what the compression step produced.
"""

import warnings

import torch
import torch.nn as nn

from ..blocks import build_block, resolve_atom
from ..blocks.stacks.config import rmsnorm_impl_from_flags
from ..compression.quantization import QUANT_RUNTIME_BUFFERS, copy_quant_state
from ..utils import flatten_index_paths, get_block_config, get_parent_module, get_submodule, replace_parameter

# Export implementation options understood by the converted stacks, with defaults.
DEFAULT_CONVERT_OPTIONS = {
    "attn_impl": None,           # None: "manual" ("sdpa" if use_sdpa), "paged" with cache_impl="io_paged"
    "use_sdpa": False,
    "cache_impl": "mutable",
    "max_cache_len": 2048,
    "add_batch_dim": False,
    "rmsnorm_impl": None,        # None: from rmsnorm_prescale / rmsnorm_upcast
    "rmsnorm_prescale": True,
    "rmsnorm_upcast": False,
    "masked_softmax": False,
}


def resolve_convert_options(options=None):
    """Fill defaults and translate the boolean RMS-normalisation flags into ``rmsnorm_impl``."""
    unknown = set(options or {}) - set(DEFAULT_CONVERT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown convert options {sorted(unknown)}; known: {sorted(DEFAULT_CONVERT_OPTIONS)}.")
    resolved = {**DEFAULT_CONVERT_OPTIONS, **(options or {})}
    prescale = resolved.pop("rmsnorm_prescale")
    upcast = resolved.pop("rmsnorm_upcast")
    if resolved["rmsnorm_impl"] is None:
        resolved["rmsnorm_impl"] = rmsnorm_impl_from_flags(prescale, upcast)
    return resolved


def _replace_param(root: nn.Module, dotted_key: str, ref: torch.Tensor) -> None:
    """Replace the parameter at ``dotted_key`` with an empty one shaped like ``ref``.

    Used to let a freshly-built block layer adopt the source layer's hard-pruned
    weight/bias shape before a strict ``load_state_dict`` copies the values in.
    """
    parent, attr = get_parent_module(root, dotted_key)
    replace_parameter(parent, attr, torch.empty_like(ref))


def _load_state_dict_dequantized(old_layer: nn.Module, new_layer: nn.Module) -> None:
    """Copy old_layer's state dict into new_layer, dequantizing quantized tensors first.

    Handles dynamic quantization buffers (_act_quant_scale, _act_quant_zero_point,
    _torchao_scale) that Quantizer.apply() registers on submodules: these appear in
    old_layer's state_dict but not in a freshly-created new_layer, so they are
    pre-registered before load_state_dict is called. The rest of the quantisation
    state is copied afterwards by ``compression.quantization.copy_quant_state``.
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
        if any(key.endswith(s) for s in QUANT_RUNTIME_BUFFERS):
            parent_path, buf_name = key.rsplit(".", 1) if "." in key else ("", key)
            parent_mod = new_modules.get(parent_path)
            if parent_mod is not None and buf_name not in parent_mod._buffers:
                parent_mod.register_buffer(buf_name, torch.empty_like(tensor))

    # Adopt the source's (possibly hard-pruned) parameter shapes, and register any
    # non-quant buffers that exist only on the source. Hard structured pruning of a
    # position-linked q_proj/k_proj shrinks its out_features and records a
    # ``rope_prune_mask`` buffer; the freshly-built block layer is still full-size
    # and lacks that buffer, so without this the strict load below fails on a
    # size mismatch / unexpected key. (Coupled dims that other layers depend on --
    # e.g. the MLP intermediate -- are pre-sized from the compression config at
    # block construction; this just reconciles the leaf tensors.)
    new_params = dict(new_layer.named_parameters())
    new_buffers = dict(new_layer.named_buffers())
    for key, tensor in plain_state.items():
        if key in new_params:
            if new_params[key].shape != tensor.shape:
                _replace_param(new_layer, key, tensor)
        elif key not in new_buffers:
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

        # in_features/out_features are plain attributes (not in the state dict), so
        # a layer that adopted hard-pruned weights above still advertises its old
        # size. Copy the source's -- the pruned ground truth (correct for LRD, where
        # in_features tracks linear_V, not weight.shape[1]) -- so downstream geometry
        # (per-head dim, RoPE projection, KV-cache sizing, out_proj input width) is
        # right.
        for dim_attr in ("in_features", "out_features"):
            if hasattr(old_submod, dim_attr):
                setattr(new_submod, dim_attr, getattr(old_submod, dim_attr))

        copy_quant_state(old_submod, new_submod)


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


def _layer_spec(module):
    """Geometry of an already-compressed linear-like source layer, or ``None`` for other layers."""
    if getattr(module, "skip", False):
        return {"out_features": 0}
    if not hasattr(module, "out_features"):
        return None
    return {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "rank": getattr(module, "rank", "full"),
        "bias": getattr(module, "bias", None) is not None,
    }


def _block_pairs(model, indexing, num_blocks):
    """``(source module path, target module path)`` for every listed layer of every block."""
    source_paths = flatten_index_paths(indexing["path_list"])
    target_paths = _flatten_layer_matching(indexing, source_paths)
    for i in range(num_blocks):
        for source, target in zip(source_paths, target_paths):
            yield i, indexing["path_template"].format(block_index=i, path=source), f"blocks.{i}.{target}"


def convert_for_export(model, options=None, verbose=False):
    """Build one export-ready stack per model component.

    Args:
        model: A compressed Hugging Face model whose class carries its model
            dictionary as ``indexing``.
        options: Export implementation options (see ``DEFAULT_CONVERT_OPTIONS``).
        verbose: Print every weight transfer.

    Returns:
        ``{component name: stack or None}``; ``None`` for components whose
        dictionary declares no ``structure``.
    """
    options = resolve_convert_options(options)
    converted = {}
    for name, indexing in model.indexing.items():
        if "structure" not in indexing:
            warnings.warn(f"Structure not specified for model with name '{name}'. Skipping conversion.")
            converted[name] = None
            continue
        if verbose:
            print(f"Converting {name} component for export...")

        source_config = get_block_config(model.config, indexing)
        num_blocks = getattr(source_config, indexing["num_blocks_attr"])
        pairs = list(_block_pairs(model, indexing, num_blocks))

        layer_specs = {}
        for _, source, target in pairs:
            spec = _layer_spec(get_submodule(model, source))
            if spec is not None:
                layer_specs[target] = spec

        stack_cls = resolve_atom("stack", indexing["structure"])
        config = stack_cls.config_class.from_source_config(source_config, indexing, layer_specs, **options)
        new_model = build_block("stack", indexing["structure"], config)

        for _, source, target in pairs:
            _load_state_dict_dequantized(get_submodule(model, source), get_submodule(new_model, target))
            if verbose:
                print("Transferring parameters:", source, "->", target)
        for i in range(num_blocks):
            # Freeze the attention geometry that depends on the loaded weights, once.
            attn = get_submodule(new_model, f"blocks.{i}.attn")
            for hook in ("finalize_head_pruning", "finalize_rope_pruning", "finalize_cache_geometry"):
                if hasattr(attn, hook):
                    getattr(attn, hook)()

        extra = indexing.get("extra_layers", [])
        extra_targets = indexing.get("extra_layers_matching") or [path.split(".")[-1] for path in extra]
        for source, target in zip(extra, extra_targets):
            if verbose:
                print("Transferring extra layer parameters:", source, "->", target)
            _load_state_dict_dequantized(get_submodule(model, source), get_submodule(new_model, target))

        converted[name] = new_model
    return converted


__all__ = ["convert_for_export", "resolve_convert_options", "DEFAULT_CONVERT_OPTIONS"]
