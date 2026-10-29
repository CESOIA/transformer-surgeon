"""Runtime manifest written next to every exported artifact.

The manifest describes the graph contract (cache atom, layouts, per-layer cache
shapes and input/output names) so runners never re-derive geometry.
"""

import json
import os
from typing import Any

import torch
import torch.nn as nn


MANIFEST_FORMAT = "tsurgeon-llm/1"


def manifest_path_for(artifact_path: str) -> str:
    """``<artifact stem>.manifest.json`` next to an exported artifact."""
    return os.path.splitext(artifact_path)[0] + ".manifest.json"


def cache_io_name(kind: str, layer: int, output: bool = False) -> str:
    """Graph I/O name of one layer's cache: ``past_<kind>_<i>`` in, ``present_<kind>_<i>`` out
    (``kind`` is ``key``, ``value`` or ``kv`` for a paged pool)."""
    return f"{'present' if output else 'past'}_{kind}_{layer}"


def graph_io_names(num_layers: int, paged: bool = False) -> tuple[list[str], list[str]]:
    """``(input_names, output_names)`` of a graph-I/O cache wrapper, in flattened argument order."""
    kinds = ("kv",) if paged else ("key", "value")
    inputs = ["input_ids", "pos_id"] + [cache_io_name(k, i) for k in kinds for i in range(num_layers)]
    outputs = ["logits"] + [cache_io_name(k, i, output=True) for k in kinds for i in range(num_layers)]
    return inputs, outputs


def build_llm_manifest(wrapper: nn.Module, model_config: Any | None, *, backend: str,
                       artifact: str, **extra: Any) -> dict[str, Any]:
    """Describe an exported LLM's runtime contract, for any backend.

    Runners use it instead of re-deriving geometry: the cache implementation,
    layout and dtype, and one entry per layer with the key/value cache shapes
    (which differ per layer after structured pruning) and their graph I/O names.
    Backend-specific fields go in ``extra``.
    """
    decoder = wrapper.decoder
    cache_impl = getattr(decoder, "cache_impl", "mutable")
    kinds = ("kv",) if cache_impl == "io_paged" else ("key", "value")
    caches = [
        {"layer": i, **{k: {"input": cache_io_name(k, i), "output": cache_io_name(k, i, output=True),
                            "shape": list(block.attn.cache_shape(k))} for k in kinds}}
        for i, block in enumerate(decoder.blocks)
    ]
    eos = getattr(model_config, "eos_token_id", None) if model_config is not None else None
    return {
        "format": MANIFEST_FORMAT,
        "backend": backend,
        "artifact": os.path.basename(artifact),
        "cache_impl": cache_impl,
        "cache_io": cache_impl != "mutable",
        "cache_inplace": cache_impl in ("io_inplace", "io_paged"),
        "cache_layout": {"io_inplace": "BHSD", "io_paged": "PAGED"}.get(cache_impl, "SHD"),
        # The allocated cache buffer, not config.dtype (None for configs without one).
        "cache_dtype": str(decoder.blocks[0].attn.key_cache.dtype).replace("torch.", ""),
        "max_cache_len": int(decoder.max_cache_len),
        "num_layers": len(decoder.blocks),
        "add_batch_dim": bool(getattr(wrapper, "add_batch_dim", False)),
        "attn_impl": getattr(decoder.blocks[0].attn, "attn_impl", None) if len(decoder.blocks) else None,
        # RoPE variant baked into the graph's cos/sin tables (indexing 'rope_type').
        "rope_type": getattr(getattr(decoder, "config", None), "rope_type", "default"),
        "vocab_size": int(getattr(model_config, "vocab_size", 0) or 0) if model_config is not None else 0,
        "eos_token_id": eos if isinstance(eos, list) else ([eos] if eos is not None else []),
        "caches": caches,
        **extra,
    }


def load_llm_manifest(artifact_path: str) -> dict[str, Any]:
    """Load the manifest written next to an exported artifact (or given directly)."""
    path = artifact_path if artifact_path.endswith(".manifest.json") else manifest_path_for(artifact_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No export manifest at '{path}'. Re-export with this version of transformer-surgeon "
            "(every backend now writes <artifact>.manifest.json)."
        )
    with open(path) as f:
        return json.load(f)


def zero_caches_from_manifest(
    manifest: dict[str, Any], device: torch.device | str = "cpu"
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-layer zero (key, value) caches with the exported shapes and dtype
    (io_paged: (kv pools, []) )."""
    dtype = getattr(torch, manifest["cache_dtype"])
    if manifest.get("cache_layout") == "PAGED":
        return [torch.zeros(layer["kv"]["shape"], dtype=dtype, device=device) for layer in manifest["caches"]], []
    keys = [torch.zeros(layer["key"]["shape"], dtype=dtype, device=device) for layer in manifest["caches"]]
    values = [torch.zeros(layer["value"]["shape"], dtype=dtype, device=device) for layer in manifest["caches"]]
    return keys, values


__all__ = ["MANIFEST_FORMAT", "manifest_path_for", "cache_io_name", "graph_io_names", "build_llm_manifest", "load_llm_manifest",
           "zero_caches_from_manifest"]
