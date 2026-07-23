# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Explicit conversion of "legacy" ViT checkpoint key names -- the layout
produced by the original HuggingFace `convert_vit_timm_to_pytorch.py` script
(`vit.encoder.layer.{i}.attention.attention.query/key/value`,
`attention.output.dense`, `intermediate.dense`, `output.dense`) -- into the
flattened `q_proj`/`k_proj`/`v_proj`/`o_proj`/`fc1`/`fc2` layout used by the
currently-installed `transformers` ViT modeling code
(`vit.layers.{i}.attention.q_proj`, ...).

`transformers` itself ships a runtime replacement for that conversion script
(`transformers.conversion_mapping`), but it only fires when the checkpoint's
keys already carry the model's `vit.` prefix. Bare-backbone checkpoints (e.g.
`google/vit-base-patch16-224-in21k`, uploaded straight from `ViTModel`) have
no such prefix, so the built-in renaming never triggers and all encoder
weights silently come back randomly initialized. Rather than depend on that
mechanism, this module computes the exact old-key -> new-key mapping ourselves
and hands it to `from_pretrained(key_mapping=...)`, so it works regardless of
whether the checkpoint carries the `vit.` prefix.
"""
import json
import os
from typing import Dict, List, Optional

_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"

# Applied in order: the attention output projection must be rewritten before
# the generic "output.dense" (MLP down-proj) rule, since its raw key also
# contains the substring "output.dense".
_LEGACY_TO_MODERN_SUBSTRINGS = [
    ("attention.attention.query", "attention.q_proj"),
    ("attention.attention.key", "attention.k_proj"),
    ("attention.attention.value", "attention.v_proj"),
    ("attention.output.dense", "attention.o_proj"),
    ("intermediate.dense", "mlp.fc1"),
    ("output.dense", "mlp.fc2"),
]


def _is_legacy_key(key: str) -> bool:
    return "encoder.layer." in key


def _rename_legacy_key(key: str) -> str:
    key = key.replace("encoder.layer.", "layers.")
    for old, new in _LEGACY_TO_MODERN_SUBSTRINGS:
        key = key.replace(old, new)
    if not key.startswith(("vit.", "classifier.", "pooler.")):
        key = f"vit.{key}"
    return key


def build_legacy_vit_key_mapping(raw_keys: List[str]) -> Dict[str, str]:
    """Given the raw parameter names found in a checkpoint file, return an
    explicit old-key -> new-key mapping covering both:
      - checkpoints produced by `convert_vit_timm_to_pytorch.py` that already
        carry the `vit.` prefix (e.g. `google/vit-base-patch16-224`)
      - bare backbone/`-in21k` checkpoints uploaded straight from `ViTModel`,
        which lack the `vit.` prefix entirely (e.g.
        `google/vit-base-patch16-224-in21k`)

    Returns `{}` for checkpoints already in the modern (`q_proj`/`layers.`)
    layout -- this is a no-op passthrough for those.
    """
    return {key: _rename_legacy_key(key) for key in raw_keys if _is_legacy_key(key)}


def _local_or_hub_file(pretrained_model_name_or_path, filename: str, **hub_kwargs) -> Optional[str]:
    subfolder = hub_kwargs.get("subfolder") or ""
    path_str = str(pretrained_model_name_or_path)
    if os.path.isdir(path_str):
        local_path = os.path.join(path_str, subfolder, filename)
        return local_path if os.path.isfile(local_path) else None

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            path_str,
            filename,
            revision=hub_kwargs.get("revision"),
            subfolder=subfolder or None,
            cache_dir=hub_kwargs.get("cache_dir"),
            token=hub_kwargs.get("token"),
        )
    except Exception:
        return None


def peek_checkpoint_keys(pretrained_model_name_or_path, **hub_kwargs) -> Optional[List[str]]:
    """Best-effort peek at a checkpoint's raw parameter names, without
    loading any tensor data (reads the safetensors index/header, or as a
    last resort a plain `pytorch_model.bin`). Returns `None` if the keys
    can't be determined this way -- callers should fall back to normal
    `from_pretrained` behavior in that case.
    """
    index_path = _local_or_hub_file(pretrained_model_name_or_path, _SAFE_WEIGHTS_INDEX_NAME, **hub_kwargs)
    if index_path is not None:
        with open(index_path) as f:
            return list(json.load(f)["weight_map"].keys())

    single_path = _local_or_hub_file(pretrained_model_name_or_path, _SAFE_WEIGHTS_NAME, **hub_kwargs)
    if single_path is not None:
        from safetensors import safe_open
        with safe_open(single_path, framework="pt") as f:
            return list(f.keys())

    bin_path = _local_or_hub_file(pretrained_model_name_or_path, _PYTORCH_WEIGHTS_NAME, **hub_kwargs)
    if bin_path is not None:
        import torch
        return list(torch.load(bin_path, map_location="meta", weights_only=True).keys())

    return None


__all__ = ["build_legacy_vit_key_mapping", "peek_checkpoint_keys"]
